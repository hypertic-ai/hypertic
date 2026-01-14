import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, PydanticDeprecationWarning, validate_arguments

from hypertic.utils.exceptions import ToolExecutionError, ToolNotFoundError


class ToolFunction:
    def __init__(
        self,
        func: Callable[..., Any],
        validated_func: Any,
        metadata: dict[str, Any],
        is_method: bool = False,
        validation_model: type[BaseModel] | None = None,
    ):
        self._func = func
        self._validated_func = validated_func
        self._metadata = metadata
        self._is_method = is_method
        self._validation_model = validation_model

        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__
        self.__qualname__ = func.__qualname__

        object.__setattr__(self, "_tool_metadata", metadata)

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        if obj is None:
            return self

        if self._is_method:

            def bound_method(**kwargs: Any) -> Any:
                if self._validation_model is not None:
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ("v__duplicate_kwargs", "args", "kwargs")}
                    validated = self._validation_model.model_validate(filtered_kwargs)
                    validated_kwargs = {k: v for k, v in validated.model_dump().items() if k not in ("v__duplicate_kwargs", "args", "kwargs")}
                else:
                    validated = self._validated_func.model.model_validate({"self": obj, **kwargs})
                    validated_kwargs = {k: v for k, v in validated.model_dump().items() if k != "self"}

                return self._validated_func(obj, **validated_kwargs)

            tool_metadata = getattr(self, "_tool_metadata", None)
            if tool_metadata is not None:
                object.__setattr__(bound_method, "_tool_metadata", tool_metadata)
            bound_method.__name__ = self.__name__
            bound_method.__doc__ = self.__doc__
            return bound_method

        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._is_method and args:
            return self._validated_func(*args, **kwargs)
        return self._validated_func(*args, **kwargs)

    def __repr__(self) -> str:
        return str(
            {
                "name": self._metadata["name"],
                "description": self._metadata["description"],
                "parameters": self._metadata["parameters"],
            }
        )


def _extract_tools_from_instance(instance: "BaseToolkit") -> list[Callable[..., Any]]:
    tools = []
    for attr_name in dir(instance):
        if attr_name.startswith("_"):
            continue

        attr = getattr(instance, attr_name)

        if isinstance(attr, ToolFunction):
            if attr._is_method:
                original_func = attr._func
                unbound_method = original_func.__func__ if hasattr(original_func, "__func__") else original_func
                bound_method = unbound_method.__get__(instance, instance.__class__)

                def make_bound_tool(
                    validation_model: type[BaseModel] | None,
                    tool_attr: ToolFunction,
                    bound_meth: Callable[..., Any],
                ) -> Callable[..., Any]:
                    def validated_bound_tool(**kwargs: Any) -> Any:
                        if validation_model is not None:
                            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ("v__duplicate_kwargs", "args", "kwargs")}
                            validated = validation_model.model_validate(filtered_kwargs)
                            validated_kwargs = {k: v for k, v in validated.model_dump().items() if k not in ("v__duplicate_kwargs", "args", "kwargs")}
                        else:
                            model = tool_attr._validated_func.model
                            field_names = [
                                name for name in model.model_fields.keys() if name not in ("self", "v__duplicate_kwargs", "args", "kwargs")
                            ]
                            validated_kwargs = {}
                            for name in field_names:
                                if name in kwargs:
                                    validated_kwargs[name] = kwargs[name]
                                elif not model.model_fields[name].is_required():
                                    default = model.model_fields[name].default
                                    if default != inspect.Parameter.empty:
                                        validated_kwargs[name] = default
                        return bound_meth(**validated_kwargs)

                    return validated_bound_tool

                bound_tool = make_bound_tool(attr._validation_model, attr, bound_method)

                if hasattr(attr, "_tool_metadata"):
                    object.__setattr__(bound_tool, "_tool_metadata", attr._tool_metadata)
                bound_tool.__name__ = attr.__name__
                bound_tool.__doc__ = attr.__doc__
                tools.append(bound_tool)
            else:
                tools.append(attr)
        elif hasattr(attr, "_tool_metadata"):

            def extract_tool(method: Callable[..., Any], metadata: dict[str, Any], inst: "BaseToolkit") -> Callable[..., Any]:
                def tool(**kwargs: Any) -> Any:
                    return method(**kwargs)

                object.__setattr__(tool, "_tool_metadata", metadata)
                tool.__name__ = method.__name__
                tool.__doc__ = method.__doc__
                return tool

            tools.append(extract_tool(attr, attr._tool_metadata, instance))

    return tools


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
    captured_name = name
    captured_description = description

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = captured_name or f.__name__
        func_description = captured_description or f.__doc__ or f"Tool: {f.__name__}"

        sig = inspect.signature(f)
        is_method = bool(f.__qualname__ and "." in f.__qualname__)
        has_self = "self" in sig.parameters
        has_cls = "cls" in sig.parameters

        validated_func: Any = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=PydanticDeprecationWarning)
                validated_func = validate_arguments(  # type: ignore
                    f, config=ConfigDict(extra="forbid", arbitrary_types_allowed=True)
                )

            model = validated_func.model

            json_schema = model.model_json_schema()

            defs = json_schema.get("$defs", {})

            valid_param_names = set(sig.parameters.keys())

            if has_self and is_method:
                valid_param_names.remove("self")
            if has_cls and is_method:
                valid_param_names.remove("cls")

            pydantic_internal_fields = {"v__duplicate_kwargs", "args", "kwargs"}

            def resolve_ref(param_info: dict[str, Any]) -> dict[str, Any]:
                """Resolve $ref references to actual schema definitions."""
                if "$ref" in param_info:
                    ref_path = param_info["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        def_name = ref_path.split("/")[-1]
                        if def_name in defs:
                            # Type ignore:
                            return defs[def_name]  # type: ignore
                return param_info

            properties = {}
            required = []

            for param_name, param_info in json_schema.get("properties", {}).items():
                if param_name in pydantic_internal_fields:
                    continue
                if param_name not in valid_param_names:
                    continue

                param_info = resolve_ref(param_info)

                if "anyOf" in param_info:
                    non_null_type = None
                    for option in param_info["anyOf"]:
                        if option.get("type") != "null":
                            non_null_type = resolve_ref(option)
                            break
                    if non_null_type:
                        param_info = non_null_type.copy()
                        if "description" in param_info:
                            pass
                        elif "title" in param_info:
                            param_info["description"] = param_info.get("title", f"Parameter: {param_name}")
                    else:
                        param_info = resolve_ref(param_info["anyOf"][0])

                cleaned_param = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description") or param_info.get("title") or f"Parameter: {param_name}",
                }

                if "items" in param_info:
                    cleaned_param["items"] = param_info["items"]
                if "additionalProperties" in param_info:
                    cleaned_param["additionalProperties"] = param_info["additionalProperties"]
                if "enum" in param_info:
                    cleaned_param["enum"] = param_info["enum"]

                if param_info.get("type") == "object" and "properties" in param_info:
                    nested_props = {}
                    for prop_name, prop_info in param_info["properties"].items():
                        nested_prop = {
                            "type": prop_info.get("type", "string"),
                            "description": prop_info.get("description") or prop_info.get("title") or f"Parameter: {prop_name}",
                        }
                        if "default" in prop_info:
                            nested_prop["default"] = prop_info["default"]
                        if "enum" in prop_info:
                            nested_prop["enum"] = prop_info["enum"]
                        if "items" in prop_info:
                            nested_prop["items"] = prop_info["items"]
                        nested_props[prop_name] = nested_prop

                    cleaned_param["properties"] = nested_props
                    if "required" in param_info:
                        cleaned_param["required"] = param_info["required"]
                    if "additionalProperties" in param_info:
                        cleaned_param["additionalProperties"] = param_info["additionalProperties"]

                properties[param_name] = cleaned_param

                if param_name in sig.parameters:
                    param = sig.parameters[param_name]
                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)

            parameters: dict[str, Any] = {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }

        except Exception:
            sig = inspect.signature(f)
            fallback_parameters: dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                fallback_parameters["properties"][param_name] = {
                    "type": "string",
                    "description": f"Parameter: {param_name}",
                }

                if param.default == inspect.Parameter.empty:
                    fallback_parameters["required"].append(param_name)

            parameters = fallback_parameters
            if validated_func is None:
                validated_func = f

        # Store metadata
        metadata = {
            "name": tool_name,
            "description": func_description,
            "parameters": parameters,
        }

        validation_model: type[BaseModel] | None = None
        if is_method and has_self and validated_func is not None:
            try:
                from pydantic import create_model

                original_model = validated_func.model
                internal_fields = {"v__duplicate_kwargs", "args", "kwargs"}
                field_names = [name for name in original_model.model_fields.keys() if name != "self" and name not in internal_fields]
                if field_names:
                    field_definitions: dict[str, Any] = {}
                    for name in field_names:
                        field_info = original_model.model_fields[name]
                        field_definitions[name] = (field_info.annotation, field_info)
                    validation_model = create_model(
                        f"{original_model.__name__}WithoutSelf",
                        **field_definitions,
                        __config__=original_model.model_config,
                    )
            except Exception:
                validation_model = None

        return ToolFunction(f, validated_func, metadata, is_method=is_method and has_self, validation_model=validation_model)

    if func is None:
        return decorator
    else:
        return decorator(func)


class BaseToolkit(BaseModel):
    """
    Base class for toolkits.

    Toolkits group related tools together and provide a standard interface
    for extracting tools.

    Inherits from BaseModel to support Pydantic field validation.

    By default, get_tools() automatically extracts all @tool decorated methods.
    Subclasses can override get_tools() for custom behavior.
    """

    model_config = ConfigDict(ignored_types=(ToolFunction,))

    def __str__(self) -> str:
        """Return a nice string representation showing all tools in the toolkit.

        When printed, shows all tools in the same format as ToolFunction.__repr__.
        """
        try:
            tools = self.get_tools()
            if not tools:
                return f"{self.__class__.__name__}(no tools)"

            tool_dicts: list[dict[str, Any]] = []
            for tool in tools:
                if hasattr(tool, "_tool_metadata"):
                    metadata = tool._tool_metadata
                    tool_dict = {
                        "name": metadata["name"],
                        "description": metadata["description"],
                        "parameters": metadata["parameters"],
                    }
                    tool_dicts.append(tool_dict)
                else:
                    tool_dicts.append(
                        {
                            "name": getattr(tool, "__name__", "unknown"),
                            "description": getattr(tool, "__doc__", ""),
                            "parameters": {},
                        }
                    )

            return str(tool_dicts)
        except Exception:
            return super().__str__()

    def __repr__(self) -> str:
        """Return a nice string representation showing all tools in the toolkit.

        When printed in a list like [toolkit], shows all tools in a readable format
        matching ToolFunction format. Returns a string that represents the tools
        without list brackets, so Python's list printing works correctly.
        """
        try:
            tools = self.get_tools()
            if not tools:
                return f"{self.__class__.__name__}(no tools)"

            tool_reprs = []
            for tool in tools:
                if hasattr(tool, "_tool_metadata"):
                    metadata = tool._tool_metadata
                    tool_dict = {
                        "name": metadata["name"],
                        "description": metadata["description"],
                        "parameters": metadata["parameters"],
                    }
                    tool_reprs.append(str(tool_dict))
                else:
                    tool_reprs.append(str(tool))

            return ", ".join(tool_reprs)
        except Exception:
            return super().__repr__()

    def get_tools(self) -> list[Callable[..., Any]]:
        """
        Get all tools in this toolkit.

        Default implementation extracts all @tool decorated methods automatically.
        Subclasses can override this method for custom tool extraction behavior.

        Returns:
            List of tool functions (decorated with @tool)
        """
        return _extract_tools_from_instance(self)


@dataclass
class _ToolManager:
    tools: list[Callable[..., Any]] | None = None
    _tools_dict: dict[str, Callable[..., Any]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.tools:
            for tool_func in self.tools:
                self.add_tool(tool_func)

    def add_tool(self, tool_func: Callable[..., Any]) -> None:
        if hasattr(tool_func, "_tool_metadata"):
            tool_name = tool_func._tool_metadata["name"]
            if tool_name in self._tools_dict:
                raise ValueError(
                    f"Duplicate tool name: '{tool_name}'. "
                    f"Tool names must be unique. "
                    f"Existing tool: {self._tools_dict[tool_name].__name__}, "
                    f"New tool: {tool_func.__name__}"
                )
            self._tools_dict[tool_name] = tool_func
        else:
            raise ValueError(f"Function {tool_func.__name__} is not decorated with @tool")

    def get_tool(self, name: str) -> Callable[..., Any] | None:
        return self._tools_dict.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools_dict.keys())

    def to_openai_format(self) -> list[dict[str, Any]]:
        openai_tools = []
        for tool_func in self._tools_dict.values():
            if hasattr(tool_func, "_tool_metadata"):
                metadata = tool_func._tool_metadata
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": metadata["name"],
                            "description": metadata["description"],
                            "parameters": metadata["parameters"],
                        },
                    }
                )
        return openai_tools

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        tool_func = self.get_tool(name)
        if not tool_func:
            raise ToolNotFoundError(f"Tool '{name}' not found")

        try:
            result = tool_func(**arguments)
            return str(result)
        except ToolExecutionError:
            raise
        except ValueError as e:
            raise ToolExecutionError(f"Error executing tool '{name}': {e}") from e
        except Exception as e:
            raise ToolExecutionError(f"Error executing tool '{name}': {e}") from e
