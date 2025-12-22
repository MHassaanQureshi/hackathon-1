from pydantic import BaseModel
from typing import List, Optional

class ModuleBase(BaseModel):
    title: str
    description: str
    order: int

class ModuleCreate(ModuleBase):
    pass

class ModuleUpdate(ModuleBase):
    pass

class ModuleResponse(ModuleBase):
    id: int

    class Config:
        from_attributes = True

class ChapterBase(BaseModel):
    title: str
    order: int
    content: str

class ChapterCreate(ChapterBase):
    module_id: int

class ChapterUpdate(ChapterBase):
    pass

class ChapterResponse(ChapterBase):
    id: int
    module_id: int

    class Config:
        from_attributes = True

class ExerciseBase(BaseModel):
    title: str
    exercise_type: str
    difficulty: str
    content: str
    solution: str

class ExerciseCreate(ExerciseBase):
    chapter_id: int

class ExerciseResponse(ExerciseBase):
    id: int
    chapter_id: int

    class Config:
        from_attributes = True

class ResourceBase(BaseModel):
    title: str
    url: str
    resource_type: str

class ResourceCreate(ResourceBase):
    pass

class ResourceResponse(ResourceBase):
    id: int

    class Config:
        from_attributes = True

class SearchResult(BaseModel):
    id: str
    title: str
    contentPreview: str
    module: str
    chapter: str
    relevanceScore: float

class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None
    sessionId: Optional[str] = None

class Source(BaseModel):
    id: str
    title: str
    module: str
    chapter: str
    relevance: float

class ChatResponse(BaseModel):
    response: str
    sources: List[Source]
    sessionId: str