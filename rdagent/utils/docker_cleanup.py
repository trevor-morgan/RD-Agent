"""
Docker resource cleanup utilities for RD-Agent.

Provides functions for cleaning up Docker resources (dangling images, stopped containers,
build cache) at various points in the workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import docker
from rdagent.log import rdagent_logger as logger

if TYPE_CHECKING:
    from docker import DockerClient


class DockerCleanupManager:
    """Manages Docker resource cleanup operations."""

    def __init__(self, client: DockerClient | None = None):
        self._client = client

    @property
    def client(self) -> DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def cleanup_dangling_images(self) -> dict:
        """Remove dangling (untagged) images.

        Returns:
            dict with SpaceReclaimed and ImagesDeleted
        """
        try:
            result = self.client.images.prune(filters={"dangling": True})
            space_reclaimed = result.get("SpaceReclaimed", 0)
            images_deleted = len(result.get("ImagesDeleted", []) or [])
            if images_deleted > 0:
                logger.info(
                    f"Cleaned {images_deleted} dangling images, reclaimed {space_reclaimed / 1024 / 1024:.2f} MB"
                )
            return result
        except Exception as e:
            logger.warning(f"Failed to cleanup dangling images: {e}")
            return {}

    def cleanup_stopped_containers(self, filters: dict | None = None) -> dict:
        """Remove stopped containers.

        Args:
            filters: Optional dict of filters (e.g., labels)

        Returns:
            dict with SpaceReclaimed and ContainersDeleted
        """
        try:
            prune_filters = filters or {}
            result = self.client.containers.prune(filters=prune_filters)
            containers_deleted = len(result.get("ContainersDeleted", []) or [])
            space_reclaimed = result.get("SpaceReclaimed", 0)
            if containers_deleted > 0:
                logger.info(
                    f"Cleaned {containers_deleted} stopped containers, reclaimed {space_reclaimed / 1024 / 1024:.2f} MB"
                )
            return result
        except Exception as e:
            logger.warning(f"Failed to cleanup stopped containers: {e}")
            return {}

    def cleanup_build_cache(self, keep_storage: str | None = None) -> dict:
        """Prune Docker build cache.

        Args:
            keep_storage: Optional size to keep (e.g., "10GB")

        Returns:
            dict with SpaceReclaimed
        """
        try:
            # Use low-level API for build cache pruning
            result = self.client.api.prune_builds(keep_storage=keep_storage)
            space_reclaimed = result.get("SpaceReclaimed", 0)
            if space_reclaimed > 0:
                logger.info(f"Cleaned build cache, reclaimed {space_reclaimed / 1024 / 1024:.2f} MB")
            return result
        except Exception as e:
            logger.warning(f"Failed to cleanup build cache: {e}")
            return {}

    def cleanup_rdagent_images(self, prefix: str = "local_") -> list[str]:
        """Remove RD-Agent specific images matching a prefix.

        Args:
            prefix: Image name prefix to match (default: "local_")

        Returns:
            List of removed image tags
        """
        removed = []
        try:
            for image in self.client.images.list():
                for tag in image.tags:
                    if tag.startswith(prefix):
                        try:
                            self.client.images.remove(image.id, force=True)
                            removed.append(tag)
                            logger.info(f"Removed image: {tag}")
                        except Exception as img_err:
                            logger.warning(f"Failed to remove image {tag}: {img_err}")
                        break
            return removed
        except Exception as e:
            logger.warning(f"Failed to cleanup RD-Agent images: {e}")
            return removed

    def get_disk_usage(self) -> dict:
        """Get Docker disk usage summary.

        Returns:
            dict with Images, Containers, Volumes, BuildCache usage info
        """
        try:
            return self.client.df()
        except Exception as e:
            logger.warning(f"Failed to get disk usage: {e}")
            return {}

    def full_cleanup(
        self,
        dangling_images: bool = True,
        stopped_containers: bool = True,
        build_cache: bool = False,
        rdagent_images: bool = False,
    ) -> dict:
        """Perform a full cleanup of Docker resources.

        Args:
            dangling_images: Clean dangling images (default: True)
            stopped_containers: Clean stopped containers (default: True)
            build_cache: Clean build cache (default: False)
            rdagent_images: Clean RD-Agent images (default: False)

        Returns:
            dict with results for each cleanup type
        """
        results = {}
        if dangling_images:
            results["dangling_images"] = self.cleanup_dangling_images()
        if stopped_containers:
            results["stopped_containers"] = self.cleanup_stopped_containers()
        if build_cache:
            results["build_cache"] = self.cleanup_build_cache()
        if rdagent_images:
            results["rdagent_images"] = self.cleanup_rdagent_images()
        return results


# Singleton for convenience
_cleanup_manager: DockerCleanupManager | None = None


def get_cleanup_manager() -> DockerCleanupManager:
    """Get the singleton cleanup manager."""
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = DockerCleanupManager()
    return _cleanup_manager


def pre_build_cleanup() -> None:
    """Cleanup to run before building images.

    Removes dangling images to free up space and prevent conflicts.
    """
    manager = get_cleanup_manager()
    manager.cleanup_dangling_images()


def post_run_cleanup() -> None:
    """Cleanup to run after container execution.

    Removes stopped containers to free up space.
    """
    manager = get_cleanup_manager()
    manager.cleanup_stopped_containers()
