import enum

class Size(enum.Enum):
    SMALL = 0.1
    MEDIUM = 0.3
    LARGE = 0.5

    @classmethod
    def get(cls, name: str) -> "Size":
        return cls[name.upper()]
