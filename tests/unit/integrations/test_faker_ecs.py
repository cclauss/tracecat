import json

import httpx
import pytest
import yaml

from tracecat.actions.integrations.faker.ecs import generate_logs_from_ecs_fields

LOG_SAMPLES_DIR_PATH = "tests/data/log_samples"
ELASTIC_INTEGRATIONS_BASE_URL = (
    "https://raw.githubusercontent.com/elastic/integrations/main/packages"
)
ELASTIC_DETECTION_RULES_BASE_URL = (
    "https://raw.githubusercontent.com/elastic/protections-artifacts/main/behavior"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,examples,detection_rules",
    [
        (
            "/crowdstrike/data_stream/alert/fields/fields.yml",
            ["tests/data/log_samples/crowdstrike/alert.json"],
            [
                "/rules/execution_potential_python_reverse_shell.toml",
                "/rules/execution_powershell_encoded_command.toml",
                "/rules/execution_suspicious_apple_script_execution.toml",
                "/rules/command_and_control_suspicious_dns_query_to_free_ssl_certificate_domains.toml",
                "/rules/impact_darkradiation_ransomware_infection.toml",
            ],
        )
    ],
)
async def test_generate_alerts_from_detection_rules(source, examples, detection_rules):
    context_description = "EQL based malicious behavior rules"

    # Load YAML schema and convert to dict
    source = f"{ELASTIC_INTEGRATIONS_BASE_URL}{source}"
    response = httpx.get(source)
    response.raise_for_status()
    ecs_fields = yaml.safe_load(response.content)

    # Load examples
    examples = []
    for example in examples:
        with open(example) as f:
            examples.append(json.load(f))

    # Load ECS detection rules TOML files
    rule_contexts = []
    async with httpx.AsyncClient(base_url=ELASTIC_DETECTION_RULES_BASE_URL) as client:
        for rule in detection_rules:
            response = await client.get(rule)
            response.raise_for_status()
            rule_contexts.append(response.text)

    # Generate logs from ECS fields
    logs = await generate_logs_from_ecs_fields(
        source=source,
        examples=examples,
        contexts=rule_contexts,
        context_description=context_description,
    )

    # Check number of logs
    assert len(logs) == len(detection_rules)

    # Check log schema
    for log in logs:
        assert all(field in log for field in ecs_fields)
        assert all(isinstance(log[field], ecs_fields[field]) for field in ecs_fields)

    # TODO: Use LLM to check if generated logs align with context
