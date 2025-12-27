import React from 'react';
import ChatInterface from './ChatInterface';
import styles from './AIBotDocumentation.module.css';

const AIBotDocumentation = () => {
  return (
    <div className={styles.container}>
      <div className={styles.documentation}>
        <h1>AI Assistant for the Book</h1>
        <p>
          This AI assistant is powered by a Retrieval-Augmented Generation (RAG) system that has access to all content in the AI-Native Book on Physical AI & Humanoid Robotics.
          It can help answer questions, explain concepts, and provide relevant information from the book.
        </p>

        <h2>How to Use</h2>
        <ul>
          <li>Ask questions about any topic covered in the book</li>
          <li>Request explanations of specific concepts</li>
          <li>Ask for examples or practical applications</li>
          <li>Request summaries of specific sections</li>
        </ul>

        <h2>Features</h2>
        <ul>
          <li><strong>Context-Aware Responses:</strong> The AI understands the context of your conversation</li>
          <li><strong>Source Attribution:</strong> Responses include references to specific modules and chapters</li>
          <li><strong>Accurate Information:</strong> Responses are grounded in the book content to prevent hallucinations</li>
        </ul>

        <h2>Tips for Best Results</h2>
        <ul>
          <li>Be specific with your questions</li>
          <li>Ask follow-up questions to dive deeper into topics</li>
          <li>Request examples if you need clarification</li>
          <li>Use the source references to explore related content in the book</li>
        </ul>
      </div>

      <div className={styles.chatSection}>
        <h2>Ask the AI Assistant</h2>
        <div className={styles.chatWrapper}>
          <ChatInterface />
        </div>
      </div>
    </div>
  );
};

export default AIBotDocumentation;