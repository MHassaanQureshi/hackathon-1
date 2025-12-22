from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import models
import schemas
from database import get_db

router = APIRouter()

@router.get("/modules", response_model=List[schemas.ModuleResponse])
def get_modules(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    modules = db.query(models.BookModule).offset(skip).limit(limit).all()
    return modules

@router.get("/modules/{module_id}", response_model=schemas.ModuleResponse)
def get_module(module_id: int, db: Session = Depends(get_db)):
    module = db.query(models.BookModule).filter(models.BookModule.id == module_id).first()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    return module

@router.get("/modules/{module_id}/chapters/{chapter_id}", response_model=schemas.ChapterResponse)
def get_chapter(module_id: int, chapter_id: int, db: Session = Depends(get_db)):
    chapter = db.query(models.Chapter).filter(
        models.Chapter.id == chapter_id,
        models.Chapter.module_id == module_id
    ).first()
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    return chapter

@router.get("/search", response_model=List[schemas.SearchResult])
def search_content(query: str, module: str = None, chapter: str = None, db: Session = Depends(get_db)):
    # This is a basic search implementation
    # In a real implementation, we would use the vector database for semantic search
    search_results = []

    # For now, return empty results - this will be implemented with Qdrant later
    return search_results

@router.post("/chat", response_model=schemas.ChatResponse)
def chat_with_book(request: schemas.ChatRequest):
    # This will be implemented with the RAG system later
    # For now, return a placeholder response
    return schemas.ChatResponse(
        response="This is a placeholder response. The RAG system will be implemented in later phases.",
        sources=[],
        sessionId=request.sessionId or "default-session"
    )