import React from 'react';
import clsx from 'clsx';
import styles from './Button.module.css';

const Button = ({ children, type = 'primary', size = 'medium', href, onClick, disabled }) => {
  const buttonType = type.toLowerCase();
  const buttonSize = size.toLowerCase();

  const buttonProps = {
    className: clsx(
      styles.button,
      styles[`button--${buttonType}`],
      styles[`button--${buttonSize}`],
      {
        [styles['button--disabled']]: disabled,
      }
    ),
  };

  if (href) {
    return (
      <a href={href} {...buttonProps}>
        {children}
      </a>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      {...buttonProps}
    >
      {children}
    </button>
  );
};

export default Button;