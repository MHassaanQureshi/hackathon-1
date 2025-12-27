import React, { useState, useRef, useEffect } from 'react';
import clsx from 'clsx';
import styles from './ChatInterface.module.css';

const ChatInterface = ({ initialMessages = [], sessionId = null }) => {
  const [messages, setMessages] = useState(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState(sessionId || `session_${Date.now()}`);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the backend API - using the backend server URL
      const response = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          sessionId: currentSessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Add bot response
      const botMessage = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot',
        sources: data.sources || [],
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, botMessage]);
      setCurrentSessionId(data.sessionId);
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error while processing your request. Please try again.',
        sender: 'bot',
        error: true,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatSources = (sources) => {
    if (!sources || sources.length === 0) return null;

    return (
      <div className={styles.sources}>
        <h4>Sources:</h4>
        <ul>
          {sources.map((source, index) => (
            <li key={index}>
              <strong>{source.title}</strong> ({source.module} - {source.chapter})
            </li>
          ))}
        </ul>
      </div>
    );
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.messagesContainer}>
        {messages.length === 0 ? (
          <div className={styles.welcomeMessage}>
            <h3>Welcome to the AI-Native Book Assistant!</h3>
            <p>Ask me anything about Physical AI & Humanoid Robotics. I can help explain concepts from the book and provide relevant information.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={clsx(
                styles.message,
                message.sender === 'user' ? styles.userMessage : styles.botMessage
              )}
            >
              <div className={styles.messageContent}>
                <div className={styles.messageText}>{message.text}</div>
                {message.sender === 'bot' && message.sources && formatSources(message.sources)}
                {message.error && (
                  <div className={styles.errorMessage}>
                    An error occurred. Please check that the backend is running.
                  </div>
                )}
              </div>
              <div className={styles.messageSender}>
                {message.sender === 'user' ? 'You' : 'Book Assistant'}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className={clsx(styles.message, styles.botMessage)}>
            <div className={styles.messageContent}>
              <div className={styles.typingIndicator}>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
            <div className={styles.messageSender}>Book Assistant</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className={styles.inputForm}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask a question about the book content..."
          className={styles.input}
          disabled={isLoading}
        />
        <button type="submit" className={styles.sendButton} disabled={isLoading}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;