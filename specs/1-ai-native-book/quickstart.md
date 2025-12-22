# Quickstart Guide: AI-Native Book on Physical AI & Humanoid Robotics

## Overview
This guide provides a step-by-step approach to setting up and running the AI-Native Book on Physical AI & Humanoid Robotics project. The system consists of a Docusaurus-based frontend with integrated RAG (Retrieval Augmented Generation) chatbot functionality.

## Prerequisites
- Node.js (v18.x or higher)
- npm or yarn package manager
- Python 3.9+ (for backend services)
- Git for version control
- An OpenAI API key
- Access to Qdrant Cloud (for vector database)

## Development Environment Setup

### 1. Clone the Repository
```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Install Dependencies
```bash
# Install frontend dependencies
cd website
npm install

# Install backend dependencies (in a separate terminal)
cd backend
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the backend directory with the following configuration:

```
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
DATABASE_URL=your_neon_postgres_connection_string
SECRET_KEY=your_secret_key_for_fastapi
DEBUG=true
```

### 4. Initialize the Backend
```bash
# Start the FastAPI backend server
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Start the Frontend Development Server
```bash
# In the website directory
cd website
npm run dev
```

The development server will be available at `http://localhost:3000` and the backend API at `http://localhost:8000`.

## Project Structure
```
project-root/
├── website/                 # Docusaurus frontend
│   ├── docs/               # Book content (modules and chapters)
│   ├── src/                # Custom components and styling
│   ├── static/             # Static assets
│   ├── docusaurus.config.js # Docusaurus configuration
│   └── package.json        # Frontend dependencies
├── backend/                # FastAPI backend
│   ├── main.py             # Main application entry point
│   ├── api/                # API route definitions
│   ├── models/             # Data models
│   ├── services/           # Business logic
│   └── requirements.txt    # Backend dependencies
├── specs/                  # Specification files
└── .env                    # Environment variables
```

## Content Creation Workflow

### 1. Adding a New Module
1. Create a new directory in `website/docs/`:
   ```
   website/docs/module-x-module-title/
   ```

2. Add a module index file:
   ```
   website/docs/module-x-module-title/index.md
   ```

3. Update `website/sidebars.js` to include the new module in navigation.

### 2. Adding a New Chapter
1. Create a chapter file in the appropriate module directory:
   ```
   website/docs/module-x-module-title/chapter-y-chapter-title.md
   ```

2. Use MDX format with appropriate frontmatter:
   ```md
   ---
   title: Chapter Title
   description: Brief description of the chapter
   sidebar_position: Y
   ---

   # Chapter Title

   Content goes here...
   ```

### 3. Adding Exercises
Exercises can be embedded directly in chapter content using custom MDX components:
```mdx
<Exercise
  type="code"
  difficulty="intermediate"
  title="Exercise Title"
>
## Instructions
Write your instructions here...

## Solution
```python
# Your solution code here
```

</Exercise>
```

## RAG System Integration

### 1. Content Indexing
The RAG system automatically indexes all content in the `docs/` directory. To trigger a re-index:

```bash
cd backend
python -m scripts.index_content
```

### 2. Testing the Chat Interface
The chat interface is integrated into each page. To test:

1. Navigate to any chapter page
2. Use the chat widget in the sidebar or bottom-right corner
3. Ask questions related to the current content

### 3. API Endpoints
Key API endpoints for the RAG system:
- `POST /api/chat` - Chat with the RAG system
- `POST /api/search` - Search across book content
- `GET /api/content/modules` - Get all modules
- `GET /api/content/modules/{moduleId}/chapters/{chapterId}` - Get specific chapter

## Deployment

### 1. GitHub Pages Deployment
The site is configured for GitHub Pages deployment via GitHub Actions. To deploy:

1. Push changes to the main branch
2. The workflow in `.github/workflows/deploy.yml` will automatically build and deploy

### 2. Backend Deployment
The backend can be deployed to any cloud provider that supports Python applications (e.g., Heroku, Railway, AWS, GCP).

## Development Commands

### Frontend Commands
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run serve        # Serve built site locally
npm run deploy       # Deploy to GitHub Pages
```

### Backend Commands
```bash
uvicorn main:app --reload    # Start development server
python -m pytest            # Run tests
python -m scripts.index_content  # Re-index content
```

## Troubleshooting

### Common Issues
1. **Content not appearing**: Ensure the sidebar configuration includes the new content
2. **Chat not working**: Verify backend API is running and API keys are correct
3. **Search not working**: Re-index content using the indexing script
4. **Build errors**: Check for syntax errors in MDX files

### Development Tips
- Use `DEBUG=true` in the backend .env for detailed error messages
- The Docusaurus live reload will automatically update when content changes
- Use the API documentation at `/docs` endpoint to test backend endpoints

## Next Steps
1. Start by creating your first module in the `website/docs/` directory
2. Add your content following the MDX format
3. Test the RAG functionality with your content
4. Deploy to GitHub Pages when ready for production