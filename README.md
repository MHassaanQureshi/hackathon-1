# AI-Native Book on Physical AI & Humanoid Robotics

A comprehensive educational platform featuring an interactive book on Physical AI systems and humanoid robot control with an integrated AI assistant powered by Retrieval-Augmented Generation (RAG).

## 🚀 Features

### Docusaurus UI Redesign
- **Enhanced Readability**: Improved typography with better fonts, spacing, and visual hierarchy
- **Simplified Navigation**: Clean navbar and sidebar with improved organization
- **Code Block Presentation**: Enhanced syntax highlighting with copy functionality
- **Consistent Design System**: Reusable MDX components for callouts, cards, and alerts
- **Theme System**: Light/dark mode with accessible color palettes
- **Responsive Design**: Mobile-first approach with optimized layouts for all devices
- **Performance Optimized**: Optimized CSS and loading strategies

### AI-Native Book Content
- **Module 1**: Robotic Nervous System (ROS 2)
  - ROS 2 fundamentals (nodes, topics, services)
  - Python agents with rclpy
  - Humanoid modeling with URDF
- **Module 2**: Digital Twin (Gazebo & Unity)
  - Physics simulation in Gazebo
  - High-fidelity environments in Unity
  - Sensor simulation (LiDAR, depth, IMU)
- **Module 3**: AI-Robot Brain (NVIDIA Isaac)
  - Isaac Sim and synthetic data
  - Isaac ROS and VSLAM
  - Navigation and path planning with Nav2
- **Module 4**: Vision-Language-Action (VLA)
  - Voice commands with OpenAI Whisper
  - LLM-based task planning
  - Capstone autonomous humanoid robot project

### RAG-Powered AI Assistant
- **Semantic Search**: Vector-based search using Qdrant for relevant content retrieval
- **Context-Aware Responses**: Maintains conversation history for contextual understanding
- **Source Attribution**: Provides references to specific modules and chapters
- **Hallucination Prevention**: Validates responses against source content
- **Multi-Module Support**: Access to content across all 4 book modules
- **Interactive Chat Interface**: Embedded chat component for seamless user experience

## 🛠️ Tech Stack

### Frontend
- **Framework**: Docusaurus v3.9.2
- **Styling**: CSS modules, custom themes
- **Components**: React-based MDX components

### Backend
- **Framework**: FastAPI
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Vector Database**: Qdrant for semantic search
- **AI Services**: OpenAI API for embeddings and completions
- **Language**: Python

## 📁 Project Structure

```
hackathon-1/
├── backend/                 # FastAPI backend with RAG system
│   ├── main.py             # Application entry point
│   ├── routers.py          # API route definitions
│   ├── rag.py              # RAG system implementation
│   ├── models.py           # Database models
│   ├── schemas.py          # Pydantic schemas
│   └── database.py         # Database configuration
├── website/                # Docusaurus documentation site
│   ├── docs/              # Book content (4 modules × 3 chapters)
│   ├── src/               # Custom components and styling
│   │   ├── components/    # React components (including ChatInterface)
│   │   ├── css/           # Custom styles
│   │   └── pages/         # Custom pages
│   ├── docusaurus.config.ts # Site configuration
│   └── sidebars.ts        # Navigation structure
└── specs/                  # SDD artifacts
    ├── 1-ai-native-book/   # AI book feature specs
    └── 1-docusaurus-ui-redesign/ # UI redesign specs
```

## 🚀 Getting Started

### Prerequisites
- Node.js (v18.x or higher)
- Python (v3.8 or higher)
- npm or yarn package manager

### Backend Setup
1. Navigate to the `backend` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   QDRANT_URL=your_qdrant_url_here (optional)
   QDRANT_API_KEY=your_qdrant_api_key_here (optional)
   DATABASE_URL=postgresql://user:password@localhost/dbname
   ```
4. Run the server: `python main.py`

### Frontend Setup
1. Navigate to the `website` directory
2. Install dependencies: `npm install`
3. Run the development server: `npm run start`

### Indexing Book Content
To index all book content for the RAG system, make a POST request to:
```
POST /api/index-all-content
```

## 🤖 Using the AI Assistant

The AI assistant is integrated throughout the book content:

1. **Interactive Chat**: Access the assistant via the "AI Assistant" link in the navbar
2. **Embedded Components**: Use the `<ChatInterface />` component in any MDX page
3. **Content-Specific Help**: Ask questions about specific modules or chapters
4. **Source References**: Responses include links to relevant book sections

## 📊 Implementation Status

### Docusaurus UI Redesign ✅ COMPLETE
- [X] Foundation Setup
- [X] Enhanced Content Readability
- [X] Simplified Navigation
- [X] Enhanced Code Block Presentation
- [X] Consistent Design System
- [X] Theme System Implementation
- [X] Responsive Design Implementation
- [X] Testing & Quality Assurance
- [X] Performance Optimization
- [X] Documentation & Handoff

### AI-Native Book Content ✅ COMPLETE
- [X] Module 1-4 Content Creation
- [X] RAG System Implementation
- [X] Frontend Integration
- [X] Testing & Quality Assurance
- [X] Deployment Setup
- [X] Polish & Cross-Cutting Concerns

## 🎯 Learning Outcomes

After completing this AI-Native Book, students will be able to:
- Design and implement robotic systems using ROS 2
- Create digital twins with Gazebo and Unity
- Implement AI-based robot control with NVIDIA Isaac
- Build vision-language-action systems for humanoid robots
- Utilize LLMs for robot task planning
- Integrate multiple AI modalities for autonomous systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Docusaurus](https://docusaurus.io/)
- Powered by [FastAPI](https://fastapi.tiangolo.com/)
- Vector search by [Qdrant](https://qdrant.tech/)
- AI services by [OpenAI](https://openai.com/)