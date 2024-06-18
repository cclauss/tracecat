"""Convert ECS fields into LLM-generated log samples."""

from typing import Any

import httpx
import yaml
from outlines import generate, prompt
from pydantic import BaseModel, Field

from tracecat.actions.etl.preprocessing import flatten_json, invert_flatten_json
from tracecat.actions.llms.model import create_model


@prompt
def one_shot(
    log_schema: dict[str, str],
    example: dict[str, Any],
    context: str | dict[str, Any],
    context_description: str,
):
    """You are an Elastic Common Schema engineer.
    You task is to create a JSON given a schema, an example, and some context about the log.
    The context is a {{ context_description }}.

    JSON Schema: {{ log_schema }}
    JSON Example: {{ example }}
    Context: {{ context }}

    Task: Create a log that aligns with the schema and context provided.
    JSON:
    """


async def generate_logs_from_ecs_fields(
    source: str,
    examples: list[dict[str, Any]],
    log_name: str,
    contexts: list[dict[str, Any]] | None = None,
    context_description: str | None = None,
    model_name: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Generate logs from ECS fields.

    Parameters
    ----------
    source : str
        URL to ECS fields YAML.
    log_name: str
        Name of the log. This is used to name the Pydantic model.
    examples : list[dict[str, Any]]
        List of examples to pass into few shot prompt.
    contexts: list[dict[str, Any]] | list[str], optional
        One log is generated for each context.
        If list of URLs, the contents of the URLs are loaded and used as contexts.
    context_description: str, optional
        Description of the context i.e. what the context represents.
        Must be provided if contexts is not None.
    model_name: str, optional
        Name of the LLM model to use.

    Returns
    -------
    list[dict[str, Any]]
        List of generated logs.
    """

    if contexts is not None and context_description is None:
        raise ValueError(
            "context_description must be provided if contexts is not None."
        )

    # Instantiate LLM model
    llm = create_model(model_name=model_name)

    # Load ECS fields yaml from URL
    async with httpx.AsyncClient() as client:
        response = await client.get(source)
        response.raise_for_status()
        ecs_schema = yaml.safe_load(response.text)[0]
        ecs_fields = ecs_schema["fields"]
        nested_jsons = nested_dict_transform(ecs_fields)
        flat_json_schema = flatten_json(nested_jsons, sep="__")
        flat_log_data_model = construct_model_from_flat_dict(log_name, flat_json_schema)

    # Flatten examples
    examples = [flatten_json(example, sep="__") for example in examples]

    # Generate flattened logs
    logs = []
    for context in contexts:
        if context.startswith("http") or context.startswith("https"):
            response = await client.get(context)
            response.raise_for_status()
            context = response.text

        prompt = one_shot(flat_json_schema, examples[0], context, context_description)
        flat_log_model: BaseModel = generate.json(llm, flat_log_data_model)(prompt)
        # Invert the flattened log
        flat_log = flat_log_model.model_dump()
        log = invert_flatten_json(flat_log, sep="__")
        logs.append(log)

    return logs


def nested_dict_transform(obj: Any):
    match obj:
        case {"name": name, "description": description}:
            return {name: description}
        case {"name": name, "fields": fields}:
            return {name: nested_dict_transform(fields)}
        case list():
            # This is a list of objects
            # Merge all the objects together
            return {k: v for obj in obj for k, v in nested_dict_transform(obj).items()}
        case _:
            return obj


def construct_model_from_flat_dict(
    log_name: str, flat_dict: dict[str, Any]
) -> type[BaseModel]:
    fields = {
        key: (str | None, Field(default=None, description=value))
        for key, value in flat_dict.items()
    }
    cls = create_model(log_name, **fields)
    return cls
