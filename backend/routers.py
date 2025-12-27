from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import models
import schemas
from database import get_db
from rag import RAGSystem
import logging

router = APIRouter()

# Initialize RAG system
rag_system = RAGSystem()

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
    # This is now implemented with the RAG system for semantic search
    search_results = rag_system.search_content(query, limit=10)

    formatted_results = []
    for result in search_results:
        metadata = result.get("metadata", {})
        formatted_results.append(schemas.SearchResult(
            id=result["id"],
            title=metadata.get("title", "Unknown"),
            contentPreview=result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
            module=metadata.get("module", "Unknown"),
            chapter=metadata.get("chapter", "Unknown"),
            relevanceScore=result["relevance_score"]
        ))

    return formatted_results

@router.post("/chat", response_model=schemas.ChatResponse)
def chat_with_book(request: schemas.ChatRequest):
    # This is now implemented with the RAG system
    return rag_system.chat_with_rag(request.message, request.sessionId)

@router.post("/index-module/{module_id}")
def index_module_content(module_id: int, db: Session = Depends(get_db)):
    """Index a specific module's content in the RAG system"""
    # Get the module
    module = db.query(models.BookModule).filter(models.BookModule.id == module_id).first()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    # Get all chapters for this module
    chapters = db.query(models.Chapter).filter(
        models.Chapter.module_id == module_id
    ).order_by(models.Chapter.order).all()

    if not chapters:
        raise HTTPException(status_code=404, detail="No chapters found for this module")

    # Index each chapter's content
    indexed_count = 0
    for chapter in chapters:
        content_id = f"module_{module_id}_chapter_{chapter.id}"
        metadata = {
            "module_id": module_id,
            "chapter_id": chapter.id,
            "module": module.title,
            "chapter": chapter.title,
            "title": f"{module.title} - {chapter.title}"
        }

        success = rag_system.index_content(content_id, chapter.content, metadata)
        if success:
            indexed_count += 1

    return {"message": f"Successfully indexed {indexed_count} chapters for module {module_id}"}

@router.post("/index-all-content")
def index_all_content(db: Session = Depends(get_db)):
    """Index all book content in the RAG system"""
    # Get all modules
    modules = db.query(models.BookModule).order_by(models.BookModule.order).all()

    if not modules:
        raise HTTPException(status_code=404, detail="No modules found")

    # Index each module
    total_indexed = 0
    for module in modules:
        # Get all chapters for this module
        chapters = db.query(models.Chapter).filter(
            models.Chapter.module_id == module.id
        ).order_by(models.Chapter.order).all()

        for chapter in chapters:
            content_id = f"module_{module.id}_chapter_{chapter.id}"
            metadata = {
                "module_id": module.id,
                "chapter_id": chapter.id,
                "module": module.title,
                "chapter": chapter.title,
                "title": f"{module.title} - {chapter.title}"
            }

            success = rag_system.index_content(content_id, chapter.content, metadata)
            if success:
                total_indexed += 1

    return {"message": f"Successfully indexed {total_indexed} chapters across all modules"}