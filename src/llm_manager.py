from __future__ import annotations

import itertools
import logging
import random

from typing import Any, ClassVar, List

import boto3

from botocore.config import Config
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_openai import AzureChatOpenAI, AzureOpenAI
from pydantic import BaseModel

from src.config import BedrockSettings, LLMRetrySettings, LlmSettings, VisionModelSettings


logger = logging.getLogger(__name__)


class LlmManager(BaseModel):
    """
    LLM manager for managing multiple Azure OpenAI instances.
    """

    llms: List[AzureChatOpenAI] = []
    settings: ClassVar[LlmSettings] = LlmSettings()
    loadBalancedLlm: LoadBalancedLlm = None

    def __init__(self) -> None:
        """Initialize the LLM manager with multiple Azure OpenAI instances."""
        super().__init__()

        # Base model configuration without proxies
        model_kwargs = {
            "api_version": self.settings.openai_api_version,
            "temperature": self.settings.llm_temperature,
            "max_retries": self.settings.llm_retry_settings.max_retries,
        }

        llm0 = AzureChatOpenAI(
            api_key=self.settings.ada_1_azure_openai_api_key,
            azure_deployment=self.settings.ada_1_azure_deployment,
            azure_endpoint=self.settings.ada_1_azure_openai_endpoint,
            **model_kwargs,
        )
        llm1 = AzureChatOpenAI(
            api_key=self.settings.ada_2_azure_openai_api_key,
            azure_deployment=self.settings.ada_2_azure_deployment,
            azure_endpoint=self.settings.ada_2_azure_openai_endpoint,
            **model_kwargs,
        )
        # Garage Week 24: Adding more LLMs
        # TODO: Remove this and use the APIM to loadbalance between LLMs with quotas later
        llm2 = AzureChatOpenAI(
            api_key=self.settings.ada_3_azure_openai_api_key,
            azure_deployment=self.settings.ada_3_azure_deployment,
            azure_endpoint=self.settings.ada_3_azure_openai_endpoint,
            **model_kwargs,
        )
        llm3 = AzureChatOpenAI(
            api_key=self.settings.ada_4_azure_openai_api_key,
            azure_deployment=self.settings.ada_4_azure_deployment,
            azure_endpoint=self.settings.ada_4_azure_openai_endpoint,
            **model_kwargs,
        )

        self.loadBalancedLlm = LoadBalancedLlm(llms=[llm0, llm1, llm2, llm3])
        self.llms = [llm0, llm1, llm2, llm3]

    def get_llm(self) -> AzureChatOpenAI:
        return self.loadBalancedLlm.get_llm()

    def get_llms(self) -> List[AzureChatOpenAI]:
        return self.llms

    def set_llm_max_tokens(self, max_tokens: int):
        """Set max tokens for all LLMs."""
        for llm in self.llms:
            llm.max_tokens = max_tokens


class VisionModel(BaseModel):
    vision_api: AzureChatOpenAI = None
    settings: ClassVar[VisionModelSettings] = VisionModelSettings()

    def __init__(self):
        super().__init__()
        self.vision_api = AzureChatOpenAI(
            api_key=self.settings.vision_openai_api_key,
            api_version=self.settings.vision_openai_api_version,
            azure_endpoint=self.settings.vision_azure_endpoint,
            azure_deployment=self.settings.vision_azure_deployment,
            max_tokens=self.settings.vision_max_tokens,
            model_name=self.settings.vision_model_name,
            temperature=self.settings.vision_temperature,
            max_retries=self.settings.llm_retry_settings.max_retries,
        )

    # Pydantic doesn't seem to know the types to handle AzureOpenAI, so we need to tell it to allow arbitrary types
    class Config:
        arbitrary_types_allowed = True


class LoadBalancedLlm(BaseModel):
    llms_cycle: Any = None

    def __init__(self, llms: List[AzureChatOpenAI]):
        super().__init__()
        self.llms_cycle = itertools.cycle(llms)

    def get_llm(self):
        return next(self.llms_cycle)

    def get_random_llm(self):
        return random.choice(self.llms)

    class Config:
        arbitrary_types_allowed = True


class BedrockModels(BaseModel):
    """
    Bedrock models available for use.
    """

    settings: ClassVar[BedrockSettings] = (
        BedrockSettings()
    )  # AWS Bedrock configuration including region, model IDs, and request parameters
    bedrock_sonnet_llm: ChatBedrock = None  # Instance of ChatBedrock for sonnet model
    bedrock_haiku_llm: ChatBedrock = None  # Instance of ChatBedrock for haiku model
    _bedrock_client = None  # Boto3 client for Bedrock runtime operations

    def __init__(self):
        super().__init__()
        self._bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            config=Config(
                region_name=self.settings.bedrock_region,
                signature_version=self.settings.bedrock_signature_version,
                retries={"max_attempts": self.settings.bedrock_max_retries, "mode": "standard"},
            ),
        )

        # Initialize models
        model_kwargs = {
            "max_tokens": self.settings.bedrock_max_tokens,
            "temperature": self.settings.bedrock_temperature,
            "top_k": self.settings.bedrock_top_k,
            "top_p": self.settings.bedrock_top_p,
            "stop_sequences": self.settings.bedrock_stop_sequences,
        }

        self.bedrock_sonnet_llm = ChatBedrock(
            client=self._bedrock_client,
            model_id=self.settings.bedrock_sonnet_model_id,
            model_kwargs=model_kwargs,
        )

        self.bedrock_haiku_llm = ChatBedrock(
            client=self._bedrock_client,
            model_kwargs=model_kwargs,
            model_id=self.settings.bedrock_haiku_model_id,
        )

    def get_sonnet_llm(self) -> ChatBedrock:
        return self.bedrock_sonnet_llm

    def get_haiku_llm(self) -> ChatBedrock:
        return self.bedrock_haiku_llm

    def set_llm_max_tokens(self, max_tokens: int):
        """Set max tokens for all LLMs."""
        self.bedrock_sonnet_llm.max_tokens = max_tokens
        self.bedrock_haiku_llm.max_tokens = max_tokens

    class Config:
        arbitrary_types_allowed = True
