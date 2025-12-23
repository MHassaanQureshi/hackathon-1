import React from 'react';
import clsx from 'clsx';
import styles from './CodeBlockWrapper.module.css';

// This component wraps the standard Docusaurus code block
// to provide additional styling and functionality
const CodeBlockWrapper = ({ children, className = '', title }) => {
  return (
    <div className={clsx(styles.codeBlockWrapper, className)}>
      {title && (
        <div className={styles.codeBlockHeader}>
          <span className={styles.codeBlockTitle}>{title}</span>
        </div>
      )}
      <div className={styles.codeBlockContent}>
        {children}
      </div>
    </div>
  );
};

export default CodeBlockWrapper;