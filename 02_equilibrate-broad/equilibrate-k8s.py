import contextlib
import logging
import pickle
import os
import pathlib
import subprocess
import sys
import time

from kubernetes import client, config
import yaml
import click

from openff.evaluator.backends.dask_kubernetes import (
    KubernetesPersistentVolumeClaim, KubernetesSecret,
    BaseDaskKubernetesBackend,
    DaskKubernetesBackend,
    PodResources
)
from openff.units import unit
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.client import RequestOptions

from openff.evaluator.backends import ComputeResources
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.client import EvaluatorClient, RequestOptions, ConnectionOptions
from openff.evaluator.server.server import EvaluatorServer
from openff.evaluator.layers.equilibration import EquilibrationProperty
from openff.evaluator.utils.observables import ObservableType

from openff.evaluator.forcefield import SmirnoffForceFieldSource

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)



def _save_script(contents: str, path: str):
    with open(path, "w") as f:
        f.write(contents)
    return path


def copy_file_to_storage(
    evaluator_backend,
    input_file,
    output_file,
):
    """
    Copy a file to the storage of a Kubernetes cluster.
    """
    n_current_workers = len(evaluator_backend._client.scheduler_info()["workers"])
 
    with open(input_file, "r") as f:
        data = f.read()
    future = evaluator_backend._client.submit(_save_script, data, output_file, resources={"notGPU": 1, "GPU": 0})
    future.result()
    logger.info(f"Copied {input_file} to {output_file}")


def wait_for_pod(
    pod_name: str,
    namespace: str,
    status: str = "Running",
    timeout: int = 1000,
    polling_interval: int = 10,
):
    """
    Wait for a pod to reach a certain status.
    """
    core_v1 = client.CoreV1Api()

    start_time = time.time()
    while time.time() - start_time < timeout:
        pod = core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        if pod.status.phase == status:
            return pod
        time.sleep(polling_interval)
    
    raise TimeoutError(f"Pod {pod_name} did not reach status {status} within {timeout} seconds.")
    


def get_pod_name(
    deployment_name: str,
    namespace: str = "openforcefield",
) -> str:
    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()
    
    # Get the deployment's labels
    deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
    deployment_labels = deployment.spec.selector.match_labels

    # List pods with the deployment's labels
    label_selector = ",".join([f"{key}={value}" for key, value in deployment_labels.items()])
    pods = core_v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector).items
    pod_name = pods[0].metadata.name.split("_")[0]
    return pod_name


def copy_from_storage(
    deployment_name,
    storage_path: str = "/evaluator-storage",
    namespace: str = "openforcefield",
    destination: str = ".",
):
    pod_name = get_pod_name(deployment_name, namespace)
    command = [
        "kubectl", "cp",
        f"{namespace}/{pod_name}:{storage_path}",
        destination,
    ]
    logger.info(f"Copying from storage to {destination}")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Copy failed: {stderr.decode()}")
    return stdout.decode()


    
@contextlib.contextmanager
def forward_port(
    deployment_name,
    namespace: str = "openforcefield",
    port: int = 8998,
):
    """
    Forward a port from a Kubernetes deployment to the local machine.

    This assumes that the deployment has at least one pod.
    """

    pod_name = get_pod_name(deployment_name, namespace)
    print(f"Pod name: {pod_name}")

    # Wait for the pod to be running
    wait_for_pod(pod_name, namespace, status="Running")
    command = [
        "kubectl", "port-forward", f"pod/{pod_name}", f"{port}:{port}",
        "-n", namespace,
    ]
    logger.info(f"Forwarding port {port} to pod {pod_name}")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the port forward to be established
    time.sleep(5)
    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"Port forward failed: {stderr.decode()}")
    try:
        yield
    finally:
        proc.terminate()



def create_pvc(
    namespace: str = "openforcefield",
    job_name: str = "lw",
    storage_class_name: str = "rook-cephfs-central",
    storage_space: unit.Quantity = 2 * unit.terabytes,
    apply_pvc: bool = True,
    timeout: int = 1000,
):
    """
    Create a persistent volume claim and deploy it.

    Possibly could be turned into a method of `KubernetesPersistentVolumeClaim`.
    """
    core_v1 = client.CoreV1Api()
    
    pvc_spec = client.V1PersistentVolumeClaimSpec(
        access_modes=["ReadWriteMany"],
        storage_class_name=storage_class_name,
        resources=client.V1ResourceRequirements(
            requests={
                "storage": f"{storage_space.to(unit.gigabytes).m}Gi",
            }
        ),
    )


    pvc_name = f"evaluator-storage-{job_name}"
    metadata = client.V1ObjectMeta(name=pvc_name)
    pvc = client.V1PersistentVolumeClaim(
        api_version="v1",
        kind="PersistentVolumeClaim",
        metadata=metadata,
        spec=pvc_spec,
    )
    if apply_pvc:
        api_response = core_v1.create_namespaced_persistent_volume_claim(
            namespace=namespace,
            body=pvc
        )
        logger.info(
            f"Created PVC {pvc.metadata.name}. State={api_response.status.phase}"
        )
    
        # wait
        end_time = time.time() + timeout
        while time.time() < end_time:
            pvc = core_v1.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
            if pvc.status.phase == "Bound":
                logger.info(f"PVC '{pvc_name}' is Bound.")
                return pvc_name
            logger.info(f"Waiting for PVC '{pvc_name}' to become Bound. Current phase: {pvc.status.phase}")
            time.sleep(5)
    return pvc_name


def create_deployment(
    calculation_backend,
    remote_script_path: str,
    remote_storage_path: str,
    env: dict = None,
    volumes: list[KubernetesPersistentVolumeClaim] = None,
    secrets: list[KubernetesSecret] = None,
    namespace: str = "openforcefield",
    job_name: str = "lw",
    port: int = 8998,
    image: str = "ghcr.io/lilyminium/openff-images:tmp-evaluator-dask-v5",
):
    """
    Create Kubernetes deployment for Evaluator server.
    """
    server_name = f"evaluator-server-{job_name}-deployment"
    apps_v1 = client.AppsV1Api()
    
    metadata = client.V1ObjectMeta(
        name=f"evaluator-server-{job_name}",
        labels={"k8s-app": server_name},
    )

    # generate volume mounts and volumes
    k8s_volume_mounts = []
    k8s_volumes = []
    
    if volumes is None:
        volumes = []
    if secrets is None:
        secrets = []
    for volume in volumes + secrets:
        k8s_volume_mounts.append(volume._to_volume_mount_k8s())
        k8s_volumes.append(volume._to_volume_k8s())

    k8s_env = {}
    if env is not None:
        assert isinstance(env, dict)
        k8s_env.update(env)

    k8s_env_objects = [
        client.V1EnvVar(name=key, value=value)
        for key, value in k8s_env.items()
    ]
    resources = calculation_backend._resources_per_worker

    command = [
        "python",
        remote_script_path,
        "--cluster-name",
        calculation_backend._cluster.name,
        "--namespace",
        calculation_backend._cluster.namespace,
        "--memory",
        str(resources._memory_limit.m_as(unit.gigabytes)),
        "--ephemeral-storage",
        str(resources._ephemeral_storage_limit.m_as(unit.gigabytes)),
        "--storage-path",
        remote_storage_path,
        "--port",
        str(port)
    ]
    logger.info(f"Command: {command}")
    
    container = client.V1Container(
        name=server_name,
        image=image,
        env=k8s_env_objects,
        command=command,
        resources=client.V1ResourceRequirements(
            requests={"cpu": "1", "memory": "4Gi"},
            limits={"cpu": "1", "memory": "4Gi"},
        ),
        volume_mounts=k8s_volume_mounts,
    )

    deployment_spec = client.V1DeploymentSpec(
        replicas=1,
        selector=client.V1LabelSelector(
            match_labels={"k8s-app": server_name}
        ),
        template=client.V1PodTemplateSpec(
            metadata=metadata,
            spec=client.V1PodSpec(
                containers=[container],
                volumes=k8s_volumes,
            )
        ),
    )

    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=metadata,
        spec=deployment_spec,
    )

    # submit
    api_response = apps_v1.create_namespaced_deployment(
        namespace=namespace,
        body=deployment,
    )
    logger.info(
        f"Created deployment {deployment.metadata.name}. State={api_response.status}"
    )
    return deployment.metadata.name
    

def equilibrate(
    dataset_path: str = "dataset.json",
    n_molecules: int = 2000,
    force_field: str = "openff-2.1.0.offxml",
    port: int = 8000
):
    # load dataset
    dataset = PhysicalPropertyDataSet.from_json(dataset_path)
    print(f"Loaded {len(dataset.properties)} properties from {dataset_path}")

    error = 50

    potential_energy = EquilibrationProperty()
    potential_energy.relative_tolerance = 0.05
    potential_energy.observable_type = ObservableType.PotentialEnergy
    potential_energy.n_uncorrelated_samples = 500

    density = EquilibrationProperty()
    density.relative_tolerance = 0.05
    density.observable_type = ObservableType.Density
    density.n_uncorrelated_samples = 500

    options = RequestOptions()
    options.calculation_layers = ["EquilibrationLayer"]
    density_schema = Density.default_equilibration_schema(
        n_molecules=n_molecules,
        error_tolerances=[potential_energy, density],
        max_iterations=60,
        error_on_failure=False,
        discard_initial_frames=10,
    )

    dhmix_schema = EnthalpyOfMixing.default_equilibration_schema(
        n_molecules=n_molecules,
        error_tolerances=[potential_energy, density],
        max_iterations=60,
        error_on_failure=False,
        discard_initial_frames=10,
    )

    options.add_schema("EquilibrationLayer", "Density", density_schema)
    options.add_schema("EquilibrationLayer", "EnthalpyOfMixing", dhmix_schema)
    
    force_field_source = SmirnoffForceFieldSource.from_path(
        force_field
    )

    client = EvaluatorClient(
        connection_options=ConnectionOptions(server_port=port)
    )

    # we first request the equilibration data
    # this can be copied between different runs to avoid re-running
    # the data is saved in a directory called "stored_data"

    request, error = client.request_estimate(
        dataset,
        force_field_source,
        options,
    )
    assert error is None, error

    # block until computation finished
    results, exception = request.results(synchronous=True, polling_interval=30)
    assert exception is None

    print(f"Equilibration complete")
    print(f"# estimated: {len(results.estimated_properties)}")
    print(f"# unsuccessful: {len(results.unsuccessful_properties)}")
    print(f"# exceptions: {len(results.exceptions)}")

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)



@click.command()
@click.option("--dataset-path", default="dataset.json", help="Path to dataset")
@click.option("--namespace", default="openforcefield", help="Kubernetes namespace")
@click.option("--job-name", default="lw-iris2", help="Job name")
@click.option("--storage-class-name", default="rook-cephfs-central", help="Storage class name")
@click.option("--storage-space", default=5000, help="Storage space")
@click.option("--memory", default=4, help="Memory")
@click.option("--ephemeral-storage", default=20, help="Ephemeral storage")
@click.option("--storage-path", default="/evaluator-storage", help="Storage path")
@click.option("--script-file", default="server-existing.py", help="Script file")
@click.option("--port", default=8998, help="Port")
@click.option("--image", default="ghcr.io/lilyminium/openff-images:tmp-evaluator-dask-v5", help="Image")
def main(
    dataset_path: str = "dataset.json",
    namespace: str = "openforcefield",
    job_name: str = "lw-iris",
    storage_class_name: str = "rook-cephfs-central",
    storage_space: unit.Quantity = 5000 * unit.gigabytes,
    memory: unit.Quantity = 8 * unit.gigabytes,
    ephemeral_storage: unit.Quantity = 20 * unit.gigabytes,
    storage_path: str = "/evaluator-storage",
    script_file: str = "server-existing.py",
    port: int = 8998,
    image: str = "ghcr.io/lilyminium/openff-images:tmp-evaluator-dask-v5",
):
    config.load_kube_config()
    core_v1 = client.CoreV1Api()

    from openff.evaluator.backends.backends import PodResources, ComputeResources


    results = None
    deployment_name = None

    if not isinstance(storage_space, unit.Quantity):
        storage_space = storage_space * unit.gigabytes

    if not isinstance(ephemeral_storage, unit.Quantity):
        ephemeral_storage = ephemeral_storage * unit.gigabytes

    if not isinstance(memory, unit.Quantity):
        memory = memory * unit.gigabytes

    # EVALUATOR_PACKAGE = "git+https://github.com/lilyminium/openff-evaluator.git@f06a38e"

    try:
        # set up storage
        pvc_name = create_pvc(
            namespace=namespace,
            job_name=job_name,
            storage_class_name=storage_class_name,
            storage_space=storage_space,
            apply_pvc=True,
        )
        volume = KubernetesPersistentVolumeClaim(
            name=pvc_name,
            mount_path=storage_path,
        )
        secret = KubernetesSecret(
            name="openeye-license",
            secret_name="oe-license-feb-2025",
            mount_path="/secrets/oe_license.txt",
            sub_path="oe_license.txt",
            read_only=True,
        )
        # create and submit KubeCluster
        cluster_name = f"evaluator-{job_name}"
        calculation_backend = DaskKubernetesBackend(
            cluster_name=cluster_name,
            gpu_resources_per_worker=PodResources(
                minimum_number_of_workers=0,
                maximum_number_of_workers=36,
                number_of_threads=1,
                memory_limit=memory,
                ephemeral_storage_limit=ephemeral_storage,
                number_of_gpus=1,
                preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
            ),
            cpu_resources_per_worker=PodResources(
                minimum_number_of_workers=0,
                maximum_number_of_workers=40,
                number_of_threads=1,
                memory_limit=memory,
                ephemeral_storage_limit=ephemeral_storage,
                number_of_gpus=0,
            ),
            image=image,
            namespace=namespace,
            env={
                "OE_LICENSE": "/secrets/oe_license.txt",
                # daemonic processes are not allowed to have children
                "DASK_DISTRIBUTED__WORKER__DAEMON": "False",
                "DASK_LOGGING__DISTRIBUTED": "debug",
                "DASK__TEMPORARY_DIRECTORY": "/evaluator-storage",
                "STORAGE_DIRECTORY": "/evaluator-storage",
                # "EXTRA_PIP_PACKAGES": f"--force-reinstall jupyterlab {EVALUATOR_PACKAGE}"
            },
            volumes=[volume],
            secrets=[secret],
            annotate_resources=True,
            cluster_kwargs={"resource_timeout": 300}
        )

        spec = calculation_backend._generate_cluster_spec()
        with open("cluster-spec.yaml", "w") as f:
            yaml.safe_dump(spec, f)
        calculation_backend.start()

        logger.info(f"Calculating with backend {calculation_backend}")

        # copy script to storage
        remote_script_file = os.path.join(storage_path, pathlib.Path(script_file).name)
        copy_file_to_storage(
            calculation_backend,
            script_file,
            remote_script_file
        )



        # create and submit deployment
        deployment_name = create_deployment(
            calculation_backend,
            remote_script_file,
            storage_path,
            volumes=[volume],
            secrets=[secret],
            namespace=namespace,
            job_name=job_name,
            port=port,
            env={
                "OE_LICENSE": "/secrets/oe_license.txt",
                # "EXTRA_PIP_PACKAGES": " --force-reinstall " + EVALUATOR_PACKAGE
            },
            image=image,
        )

        # run fit
        with forward_port(
            deployment_name,
            namespace=namespace,
            port=port,
        ):
            equilibrate(
                dataset_path=dataset_path,
                n_molecules=1000,
                force_field="openff-2.1.0.offxml",
                port=port
            )

    except (Exception, BaseException) as e:
        print(e)
        raise e

    finally:
        # copy over data first from pvc to local
        if deployment_name:
            copy_from_storage(
                deployment_name,
                storage_path=storage_path,
                namespace=namespace,
                destination="."
            )



            print(f"Cleaning up")
            # clean up deployment
            apps_v1 = client.AppsV1Api()
            apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
            )

        # clean up pvc
        # note this may fail if you have another pod looking at the storage
        # core_v1.delete_namespaced_persistent_volume_claim(
        #     name=pvc_name,
        #     namespace=namespace,
        # )
        

    print(results)

    


if __name__ == "__main__":
    main()

