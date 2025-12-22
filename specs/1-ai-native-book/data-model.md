# Data Model: AI-Native Book on Physical AI & Humanoid Robotics

**Feature**: 1-ai-native-book
**Created**: 2025-12-21
**Status**: Draft

## Entity: BookModule

**Description**: Represents one of the four main content modules in the book (Robotic Nervous System, Digital Twin, AI-Robot Brain, Vision-Language-Action)

**Fields**:
- `id` (string, required): Unique identifier for the module (e.g., "module-1", "module-2")
- `title` (string, required): Display title of the module
- `description` (string, required): Brief description of the module content
- `order` (integer, required): Sequential order of the module in the curriculum (1-4)
- `slug` (string, required): URL-friendly identifier for the module
- `learningObjectives` (array of strings, required): List of learning objectives for the module
- `prerequisites` (array of strings, optional): Module IDs required before this module
- `createdAt` (datetime, required): Timestamp of module creation
- `updatedAt` (datetime, required): Timestamp of last module update

**Relationships**:
- One-to-many with Chapter (one module contains many chapters)
- Many-to-many with Resource (modules can reference multiple resources)

**Validation Rules**:
- `id` must follow pattern: "module-[1-4]"
- `order` must be between 1 and 4
- `title` must not exceed 100 characters
- `description` must not exceed 500 characters
- `learningObjectives` must contain at least 1 objective

## Entity: Chapter

**Description**: Represents one of the twelve chapters (3 per module) with specific learning objectives and content

**Fields**:
- `id` (string, required): Unique identifier for the chapter (e.g., "module-1-chapter-1")
- `title` (string, required): Display title of the chapter
- `moduleId` (string, required): Reference to the parent module
- `order` (integer, required): Sequential order of the chapter within the module (1-3)
- `slug` (string, required): URL-friendly identifier for the chapter
- `content` (string, required): MDX content of the chapter
- `learningObjectives` (array of strings, required): List of learning objectives for the chapter
- `estimatedReadingTime` (integer, required): Estimated reading time in minutes
- `exercises` (array of objects, optional): Array of exercises for the chapter
- `resources` (array of strings, optional): IDs of related resources
- `prerequisites` (array of strings, optional): Chapter IDs required before this chapter
- `createdAt` (datetime, required): Timestamp of chapter creation
- `updatedAt` (datetime, required): Timestamp of last chapter update

**Relationships**:
- Many-to-one with BookModule (many chapters belong to one module)
- One-to-many with Exercise (one chapter contains many exercises)
- Many-to-many with Resource (chapters can reference multiple resources)

**Validation Rules**:
- `id` must follow pattern: "module-[1-4]-chapter-[1-3]"
- `moduleId` must reference an existing module
- `order` must be between 1 and 3
- `title` must not exceed 100 characters
- `content` must be valid MDX format
- `learningObjectives` must contain at least 1 objective
- `estimatedReadingTime` must be between 5 and 120 minutes

## Entity: Exercise

**Description**: Represents practice exercises and activities within chapters to reinforce learning

**Fields**:
- `id` (string, required): Unique identifier for the exercise
- `chapterId` (string, required): Reference to the parent chapter
- `title` (string, required): Display title of the exercise
- `type` (string, required): Type of exercise (code, theory, simulation, quiz)
- `difficulty` (string, required): Difficulty level (beginner, intermediate, advanced)
- `description` (string, required): Detailed description of the exercise
- `instructions` (string, required): Step-by-step instructions for the exercise
- `solution` (string, optional): Solution or expected outcome for the exercise
- `validationCriteria` (array of strings, optional): Criteria for validating exercise completion
- `estimatedCompletionTime` (integer, optional): Estimated time to complete in minutes
- `resources` (array of strings, optional): IDs of resources needed for the exercise
- `createdAt` (datetime, required): Timestamp of exercise creation
- `updatedAt` (datetime, required): Timestamp of last exercise update

**Relationships**:
- Many-to-one with Chapter (many exercises belong to one chapter)
- Many-to-many with Resource (exercises can reference multiple resources)

**Validation Rules**:
- `type` must be one of: "code", "theory", "simulation", "quiz"
- `difficulty` must be one of: "beginner", "intermediate", "advanced"
- `title` must not exceed 100 characters
- `description` must not exceed 2000 characters
- `solution` must be provided for simulation and code exercises

## Entity: Resource

**Description**: Represents external resources, links, and materials referenced in modules and chapters

**Fields**:
- `id` (string, required): Unique identifier for the resource
- `title` (string, required): Display title of the resource
- `url` (string, required): URL or path to the resource
- `type` (string, required): Type of resource (video, code, paper, tool, documentation)
- `description` (string, optional): Brief description of the resource
- `relatedModuleIds` (array of strings, optional): IDs of related modules
- `relatedChapterIds` (array of strings, optional): IDs of related chapters
- `tags` (array of strings, optional): Topic tags for the resource
- `createdAt` (datetime, required): Timestamp of resource creation
- `updatedAt` (datetime, required): Timestamp of last resource update

**Relationships**:
- Many-to-many with BookModule (resources can be related to multiple modules)
- Many-to-many with Chapter (resources can be related to multiple chapters)

**Validation Rules**:
- `type` must be one of: "video", "code", "paper", "tool", "documentation"
- `title` must not exceed 100 characters
- `url` must be a valid URL or relative path
- `tags` array must not exceed 10 tags

## Entity: ChatSession

**Description**: Represents a user's chat session with the RAG system for tracking conversation history

**Fields**:
- `id` (string, required): Unique identifier for the chat session
- `userId` (string, optional): Identifier for the user (if tracking)
- `moduleId` (string, optional): Module context for the session
- `chapterId` (string, optional): Chapter context for the session
- `createdAt` (datetime, required): Timestamp of session creation
- `updatedAt` (datetime, required): Timestamp of last session update
- `messages` (array of objects, required): Array of messages in the session

**Relationships**:
- One-to-many with ChatMessage (one session contains many messages)

**Validation Rules**:
- `messages` array must not exceed 50 messages without summary
- Either `moduleId` or `chapterId` or both can be set for context
- Session must be updated within 24 hours or be considered expired

## Entity: ChatMessage

**Description**: Represents individual messages within a chat session

**Fields**:
- `id` (string, required): Unique identifier for the message
- `sessionId` (string, required): Reference to the parent chat session
- `role` (string, required): Role of the message sender (user, assistant)
- `content` (string, required): Content of the message
- `timestamp` (datetime, required): When the message was created
- `sources` (array of objects, optional): Sources referenced in the response
- `feedback` (string, optional): User feedback on the response (positive, negative)

**Relationships**:
- Many-to-one with ChatSession (many messages belong to one session)

**Validation Rules**:
- `role` must be either "user" or "assistant"
- `content` must not exceed 5000 characters
- `sources` must reference actual content from the book

## Entity: UserProgress

**Description**: Tracks user progress through the book modules and chapters

**Fields**:
- `id` (string, required): Unique identifier for the progress record
- `userId` (string, required): Identifier for the user
- `moduleId` (string, required): Reference to the module
- `chapterId` (string, optional): Reference to the specific chapter
- `status` (string, required): Progress status (not-started, in-progress, completed)
- `completionPercentage` (number, required): Percentage of content completed
- `timeSpent` (integer, optional): Time spent on the content in seconds
- `lastAccessedAt` (datetime, required): When the content was last accessed
- `exercisesCompleted` (array of strings, optional): IDs of completed exercises
- `createdAt` (datetime, required): Timestamp of progress record creation
- `updatedAt` (datetime, required): Timestamp of last progress update

**Relationships**:
- Many-to-one with BookModule (many progress records for one module)
- Many-to-one with Chapter (many progress records for one chapter)

**Validation Rules**:
- `status` must be one of: "not-started", "in-progress", "completed"
- `completionPercentage` must be between 0 and 100
- `timeSpent` must be non-negative

## Data Relationships Summary

```
BookModule (1) ←→ (many) Chapter
Chapter (1) ←→ (many) Exercise
Chapter (1) ←→ (many) ChatMessage
BookModule (many) ←→ (many) Resource
Chapter (many) ←→ (many) Resource
ChatSession (1) ←→ (many) ChatMessage
ChatSession (1) ←→ (many) UserProgress
BookModule (1) ←→ (many) UserProgress
Chapter (1) ←→ (many) UserProgress
```

## State Transitions

### Chapter Status Transitions
- `not-started` → `in-progress` (when user starts reading)
- `in-progress` → `completed` (when user completes reading and exercises)
- `completed` → `in-progress` (if user returns to review)

### Chat Session Transitions
- `active` → `inactive` (after 30 minutes of inactivity)
- `inactive` → `archived` (after 24 hours or manual user action)

## Indexing Strategy

### Primary Indexes
- BookModule.id (unique)
- Chapter.id (unique)
- Exercise.id (unique)
- Resource.id (unique)
- ChatSession.id (unique)
- ChatMessage.id (unique)
- UserProgress.id (unique)

### Secondary Indexes
- Chapter.moduleId (for module-based queries)
- Chapter.slug (for URL routing)
- Resource.tags (for tag-based search)
- ChatSession.userId (for user session queries)
- UserProgress.userId (for user progress tracking)
- UserProgress.moduleId (for module progress queries)