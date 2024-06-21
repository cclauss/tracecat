"""Convert ECS fields into LLM-generated log samples."""

import json
from typing import Annotated, Any

import httpx
import yaml
from outlines import generate, prompt

from tracecat.actions.etl.preprocessing import flatten_json
from tracecat.actions.llms.model import create_model
from tracecat.registry import Field, registry


def convert_records_to_kv_json(obj: dict | list) -> dict[str, Any]:
    match obj:
        case {"name": name, "description": description}:
            return {name: description}
        case {"name": name, "fields": fields}:
            return {name: convert_records_to_kv_json(fields)}
        case list():
            # This is a list of objects
            # We assume homogenous list and convert to a single object
            return {
                k: v for obj in obj for k, v in convert_records_to_kv_json(obj).items()
            }
        case _:
            return obj


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


async def _generate_logs_from_ecs_fields(
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
        kv_ecs_fields = convert_records_to_kv_json(ecs_fields)
        print(json.dumps(kv_ecs_fields, indent=2))
        flat_json_schema = flatten_json(kv_ecs_fields, sep=".")

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
        log = generate.json(llm)(prompt)
        logs.append(log)

    return logs


@registry.register(
    default_title="Generate Synthetic Logs (via ECS Fields)",
    description="Generate synthetic logs from ECS fields",
    display_group="Detection Engineering",
    namespace="integrations.detection_engineering.ecs.generate_sythentic_logs",
)
async def generate_logs_from_ecs_fields(
    source: Annotated[str, Field(..., description="URL to ECS fields YAML.")],
    examples: Annotated[
        list[dict[str, Any]],
        Field(..., description="List of examples to pass into few shot prompt."),
    ],
    log_name: Annotated[
        str,
        Field(
            ..., description="Name of the log. This is used to name the Pydantic model."
        ),
    ],
    contexts: Annotated[
        list[dict[str, Any]] | None,
        Field(
            None,
            description="One log is generated for each context. If list of URLs, the contents of the URLs are loaded and used as contexts.",
        ),
    ] = None,
    context_description: Annotated[
        str | None,
        Field(
            None,
            description="Description of the context i.e. what the context represents. Must be provided if contexts is not None.",
        ),
    ] = None,
    model_name: Annotated[
        str, Field(default="gpt-4o", description="Name of the LLM model to use.")
    ] = "gpt-4o",
) -> list[dict[str, Any]]:
    return await _generate_logs_from_ecs_fields(
        source=source,
        examples=examples,
        log_name=log_name,
        contexts=contexts,
        context_description=context_description,
        model_name=model_name,
    )
