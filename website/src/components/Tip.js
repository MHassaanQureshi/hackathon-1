import React from 'react';
import clsx from 'clsx';
import styles from './Tip.module.css';

const TipIcon = () => (
  <svg
    className={styles.tipIcon}
    fill="currentColor"
    viewBox="0 0 24 24"
    width="24"
    height="24"
  >
    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2V7zm0 8h2v2h-2v-2z" />
  </svg>
);

const Tip = ({ children, type = 'tip' }) => {
  const tipType = type.toLowerCase();
  const icon = <TipIcon />;

  return (
    <div className={clsx(styles.tip, styles[`tip--${tipType}`])}>
      <div className={styles.tipHeading}>
        <div className={styles.tipIcon}>{icon}</div>
        <h5 className={styles.tipTitle}>
          {tipType === 'note' ? 'Note' :
           tipType === 'tip' ? 'Tip' :
           tipType === 'caution' ? 'Caution' :
           tipType === 'danger' ? 'Danger' : 'Info'}
        </h5>
      </div>
      <div className={styles.tipContent}>{children}</div>
    </div>
  );
};

export default Tip;