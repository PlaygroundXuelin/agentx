import argparse
from dataclasses import is_dataclass
from pathlib import Path
from typing import Final, Sequence, TypeVar

import structlog
from dotenv import load_dotenv

import core.settings

LOGGER: Final = structlog.get_logger(__name__)

T = TypeVar("T")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for configuring the API entry point."""

    parser = argparse.ArgumentParser(description="Run the Agentic RAG API server.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Path to an env file to load before parsing configuration.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        parser.error(f"Configuration file '{config_path}' does not exist.")
    args.config = config_path

    if args.env is not None:
        env_path = Path(args.env).expanduser()
        if not env_path.is_file():
            parser.error(f"Env file '{env_path}' does not exist.")
        args.env = env_path

    return args


def _load_app_settings(
    dataclass_type: type[T],
    config_path: str | Path,
    env_path: str | Path | None = None,
) -> T:
    """Hydrate application settings from YAML and environment variables."""

    LOGGER.debug(
        "Loading application settings from YAML",
        config_path=config_path,
        env_path=env_path,
    )

    if env_path is not None:
        env_path = Path(env_path).expanduser()
        if not env_path.is_file():
            msg = f"Env file '{env_path}' does not exist."
            raise FileNotFoundError(msg)
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv()

    resolved_config_path = Path(config_path).expanduser()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(resolved_config_path)
    return core.settings.load_dataclass_from_yaml(dataclass_type, resolved_config_path)


def load_app_settings(
    dataclass_type: type[T],
    argv: Sequence[str] | None = None,
) -> T:
    """Convenience helper for constructing the API app."""

    args = _parse_args(argv)

    LOGGER.debug("parsed args", args=args, argv=argv)
    if not is_dataclass(dataclass_type):
        msg = f"Expected a dataclass type, got {dataclass_type!r}"
        raise TypeError(msg)
    return _load_app_settings(
        dataclass_type, config_path=args.config, env_path=args.env
    )
