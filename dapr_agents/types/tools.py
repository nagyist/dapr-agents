from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, field_validator, ValidationInfo
from datetime import timedelta


class OAIFunctionDefinition(BaseModel):
    """
    Represents a callable function in the OpenAI API format.

    Attributes:
        name (str): The name of the function.
        description (str): A detailed description of what the function does.
        parameters (Dict): A dictionary describing the parameters that the function accepts.
    """

    name: str
    description: str
    parameters: Dict


class OAIToolDefinition(BaseModel):
    """
    Represents a tool (callable function) in the OpenAI API format. This can be a function, code interpreter, or file search tool.

    Attributes:
        type (Literal["function", "code_interpreter", "file_search"]): The type of the tool.
        function (Optional[OAIBaseFunctionDefinition]): The function definition, required if type is 'function'.
    """

    type: Literal["function", "code_interpreter", "file_search"]
    function: Optional[OAIFunctionDefinition] = None

    @field_validator("function")
    def check_function_requirements(cls, v, info: ValidationInfo):
        if info.data.get("type") == "function" and not v:
            raise ValueError(
                "Function definition must be provided for function type tools."
            )
        return v


class ClaudeToolDefinition(BaseModel):
    """
    Represents a tool (callable function) in the Anthropic's Claude API format, suitable for integration with Claude's API services.

    Attributes:
        name (str): The name of the function.
        description (str): A description of the function's purpose and usage.
        input_schema (Dict): A dictionary defining the input schema for the function.
    """

    name: str
    description: str
    input_schema: Dict


class GeminiFunctionDefinition(BaseModel):
    """
    Represents a callable function in the Google's Gemini API format.

    Attributes:
        name (str): The name of the function to call. Must start with a letter or an underscore. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
        description (str): The description and purpose of the function. The model uses this to decide how and whether to call the function. For the best results, we recommend that you include a description.
        parameters (Dict): Describes the parameters of the function in the OpenAPI JSON Schema Object format: OpenAPI 3.0 specification.
    """

    name: str
    description: str
    parameters: Dict


class GeminiToolDefinition(BaseModel):
    """
    Represents a tool (callable function) in the Google's Gemini API format, suitable for integration with Gemini's API services.

    Attributes:
        function_declarations (List): A structured representation of a function declaration as defined by the OpenAPI 3.0 specification that represents a function the model may generate JSON inputs for.
    """

    function_declarations: List[GeminiFunctionDefinition]


class SseServerParameters(BaseModel):
    """
    Configuration for Server-Sent Events (SSE) transport.

    Attributes:
        url (str): The SSE endpoint URL.
        headers (Optional[Dict[str, str]]): Optional HTTP headers.
        timeout (float): Connection timeout in seconds.
        sse_read_timeout (float): Timeout for SSE read operations.
    """

    url: str
    headers: Optional[Dict[str, str]] = None
    timeout: float = 5.0
    sse_read_timeout: float = 300.0


class StreamableHTTPServerParameters(BaseModel):
    """
    Configuration for streamable HTTP transport.

    Attributes:
        url (str): The streamable HTTP endpoint URL.
        headers (Optional[Dict[str, str]]): Optional HTTP headers.
        timeout (timedelta): Connection timeout as a timedelta.
        sse_read_timeout (timedelta): Timeout for SSE read operations as a timedelta.
        terminate_on_close (bool): Whether to terminate the connection on close.
    """

    url: str
    headers: Optional[Dict[str, str]] = None
    timeout: timedelta = timedelta(seconds=30)
    sse_read_timeout: timedelta = timedelta(seconds=300)
    terminate_on_close: bool = True


class WebSocketServerParameters(BaseModel):
    """
    Configuration for websocket transport.

    Attributes:
        url (str): The websocket endpoint URL.
    """

    url: str
