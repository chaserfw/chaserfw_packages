from typing import Any


class SingletonMeta(type):
    __m_instances = {}
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls.__m_instances:
            instance = super().__call__(*args, **kwargs)
            cls.__m_instances[cls] = instance
        return cls.__m_instances[cls]