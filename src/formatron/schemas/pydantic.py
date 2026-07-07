"""
A module that implements the Schema interface using pydantic.
"""
import inspect
import json
import typing

import pydantic.fields
from pydantic import BaseModel, validate_call, ConfigDict, Field

import formatron.schemas.schema as schema
from formatron.schemas.schema import Schema, TypeWithMetadata


class FieldInfo(schema.FieldInfo):
    """
    A wrapper for pydantic FieldInfo.
    """
    __slots__ = ("_field",)

    def __init__(self, field: pydantic.fields.FieldInfo):
        """
        Initialize the field information.
        """
        self._field = field
        self._annotation = field.annotation
        if field.metadata:
            metadata = {}
            for constraint in ["min_length", "max_length", "pattern", "gt", "ge", "lt", "le", "substring_of"]:
                value = next((getattr(m, constraint) for m in self._field.metadata if hasattr(m, constraint)), None)
                if value is not None:
                    metadata[constraint] = value
            if metadata:
                self._annotation = TypeWithMetadata(self._annotation, metadata)

    @property
    def annotation(self) -> typing.Type[typing.Any] | None:
        return self._annotation

    @property
    def required(self) -> bool:
        return self._field.is_required()

    def __repr__(self):
        return repr(self._field)

    def __str__(self):
        return str(self._field)


class ClassSchema(BaseModel, Schema):
    """
    A wrapper for pydantic BaseModel that implements the Schema interface.
    """
    __cached_fields__ = None

    @classmethod
    def fields(cls) -> typing.Dict[str, typing.Any]:
        if cls.__cached_fields__ is not None:
            return cls.__cached_fields__
        cls.__cached_fields__ = {k: FieldInfo(v) for k, v in cls.model_fields.items()}
        return cls.__cached_fields__

    @classmethod
    def from_json(cls, _json: str) -> "ClassSchema":
        """
        Create a ClassSchema from a JSON string.
        """
        return cls.model_validate_json(_json)


CallableT = typing.TypeVar('CallableT', bound=typing.Callable)


def callable_schema(func: CallableT, /, *, config: ConfigDict = None, validate_return: bool = False) -> CallableT:
    """
    A decorator that wraps pydantic's validate_call. The decorated callable also implements the Schema interface.

    Args:
        func: The function to decorate.
        config: The pydantic configuration of validate_call.
        validate_return: Whether to validate the return value.

    Returns:
        The decorated callable.
    """
    pydantic_wrapper = validate_call(config=config, validate_return=validate_return)(func)
    signature = inspect.signature(func, eval_str=True)
    fields = {}
    for k, p in signature.parameters.items():
        python_default = inspect.Signature.empty
        if p.default is not inspect.Signature.empty and not isinstance(p.default, pydantic.fields.FieldInfo):
            python_default = p.default
        actual_type = p.annotation
        metadata = []
        fieldinfo = None
        if isinstance(p.default, pydantic.fields.FieldInfo):
            fieldinfo = p.default
        if typing.get_origin(p.annotation) is typing.Annotated:
            actual_type, *meta = typing.get_args(p.annotation)
            for i in meta:
                if isinstance(i, pydantic.fields.FieldInfo):
                    fieldinfo = i
                else:
                    metadata.append(i)
        if fieldinfo is None:
            if python_default is not inspect.Signature.empty:
                fieldinfo = Field(python_default)
            else:
                fieldinfo = Field()
        if python_default is not inspect.Signature.empty:
            fieldinfo.default = python_default
        fieldinfo.annotation = actual_type
        fieldinfo.metadata.extend(metadata)
        fields[k] = fieldinfo
    for k in fields:
        fields[k] = FieldInfo(fields[k])

    def from_json(cls, json_str):
        json_data = json.loads(json_str)
        positional_only = []
        others = {}
        skipping_positional_only = False
        for k, p in signature.parameters.items():
            if p.kind == p.VAR_POSITIONAL:
                if k in json_data:
                    value = json_data[k]
                    if not isinstance(value, list):
                        raise TypeError(f"{k} must be a JSON array")
                    positional_only.extend(value)
                continue
            if p.kind == p.VAR_KEYWORD:
                if k in json_data:
                    value = json_data[k]
                    if not isinstance(value, dict):
                        raise TypeError(f"{k} must be a JSON object")
                    others.update(value)
                continue
            required = fields[k].required
            if p.kind == p.POSITIONAL_ONLY:
                if k in json_data:
                    if skipping_positional_only:
                        raise ValueError(f"cannot specify {k} without earlier positional-only parameters")
                    positional_only.append(json_data[k])
                else:
                    if required:
                        raise KeyError(k)
                    skipping_positional_only = True
            else:
                if k in json_data:
                    others[k] = json_data[k]
                else:
                    if required:
                        raise KeyError(k)
        return cls(*positional_only, **others)

    _class = type(
        f'{func.__qualname__}_PydanticSchema_{id(func)}',
        (Schema,),
        {
            "_func": pydantic_wrapper,
            '__new__': lambda cls, *args, **kwargs: pydantic_wrapper(*args, **kwargs),
            '__call__': lambda *args, **kwargs: pydantic_wrapper(*args, **kwargs)  # make duck typer happy
        }
    )
    _class.from_json = classmethod(from_json)
    _class.fields = classmethod(lambda cls: fields)
    return _class
