import React from 'react';
import clsx from 'clsx';
import styles from './Callout.module.css';

const Callout = ({ children, type = 'default', title }) => {
  const calloutType = type.toLowerCase();

  return (
    <div className={clsx(styles.callout, styles[`callout--${calloutType}`])}>
      {title && <div className={styles.calloutTitle}>{title}</div>}
      <div className={styles.calloutContent}>{children}</div>
    </div>
  );
};

export default Callout;