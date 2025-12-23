import React from 'react';
import clsx from 'clsx';
import styles from './Card.module.css';

const Card = ({ title, children, type = 'default' }) => {
  return (
    <div className={clsx(styles.card, styles[`card--${type}`])}>
      {title && <h3 className={styles.cardTitle}>{title}</h3>}
      <div className={styles.cardContent}>{children}</div>
    </div>
  );
};

export default Card;