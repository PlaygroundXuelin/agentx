import pydantic.dataclasses as pydantic_dataclasses


@pydantic_dataclasses.dataclass(frozen=True)
class LocalObjectStoreSettings:
    path: str = "/tmp/_documents"


@pydantic_dataclasses.dataclass(frozen=True)
class ObjectStoreSettings:
    type: str = "LOCAL"  # must be one of: LOCAL
    settings: LocalObjectStoreSettings = LocalObjectStoreSettings()


@pydantic_dataclasses.dataclass(frozen=True)
class EmbedSettings:
    model_name: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 384  # may use llamaindex's default instead


@pydantic_dataclasses.dataclass(frozen=True)
class DocumentSettings:
    store: ObjectStoreSettings = ObjectStoreSettings()
    summary_model_name: str = "openai/gpt-4o-mini"
    embed: EmbedSettings = EmbedSettings()
