import React, { useEffect, useState } from 'react';
import clsx from 'clsx';
import styles from './ThemeToggle.module.css';

const ThemeToggle = () => {
  const [theme, setTheme] = useState('auto');

  useEffect(() => {
    // Check user's preference from localStorage or system preference
    const savedTheme = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme) {
      setTheme(savedTheme);
      applyTheme(savedTheme, systemPrefersDark);
    } else {
      setTheme('auto');
      applyTheme('auto', systemPrefersDark);
    }
  }, []);

  const applyTheme = (selectedTheme, systemPrefersDark) => {
    let themeToApply = selectedTheme;

    if (selectedTheme === 'auto') {
      themeToApply = systemPrefersDark ? 'dark' : 'light';
    }

    // Apply theme to document
    document.documentElement.setAttribute('data-theme', themeToApply);
    localStorage.setItem('theme', selectedTheme);
  };

  const toggleTheme = () => {
    let newTheme;
    if (theme === 'auto') {
      // If current is auto, switch to system opposite
      const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      newTheme = systemPrefersDark ? 'light' : 'dark';
    } else if (theme === 'light') {
      newTheme = 'dark';
    } else {
      newTheme = 'light';
    }

    setTheme(newTheme);
    applyTheme(newTheme, window.matchMedia('(prefers-color-scheme: dark)').matches);
  };

  const setAutoTheme = () => {
    setTheme('auto');
    applyTheme('auto', window.matchMedia('(prefers-color-scheme: dark)').matches);
  };

  return (
    <div className={styles.themeToggle}>
      <div className={styles.themeToggleButtons}>
        <button
          className={clsx(
            styles.themeButton,
            theme === 'light' && styles.themeButtonActive
          )}
          onClick={() => {
            setTheme('light');
            applyTheme('light', false);
          }}
          aria-label="Switch to light theme"
          title="Light theme"
        >
          <svg
            className={styles.themeIcon}
            viewBox="0 0 24 24"
            width="18"
            height="18"
          >
            <path
              fill="currentColor"
              d="M12,7c-2.8,0-5,2.2-5,5s2.2,5,5,5s5-2.2,5-5S14.8,7,12,7L12,7z M2,13l2,0c0.6,0,1-0.4,1-1s-0.4-1-1-1l-2,0 c-0.6,0-1,0.4-1,1S1.4,13,2,13z M20,13l2,0c0.6,0,1-0.4,1-1s-0.4-1-1-1l-2,0c-0.6,0-1,0.4-1,1S19.4,13,20,13z M11,2v2 c0,0.6,0.4,1,1,1s1-0.4,1-1V2c0-0.6-0.4-1-1-1S11,1.4,11,2z M11,20v2c0,0.6,0.4,1,1,1s1-0.4,1-1v-2c0-0.6-0.4-1-1-1S11,19.4,11,20z M5.6,6.6 l1.4,1.4c0.4,0.4,1,0.4,1.4,0c0.4-0.4,0.4-1,0-1.4L7,5.2c-0.4-0.4-1-0.4-1.4,0C5.2,5.6,5.2,6.2,5.6,6.6z M18.4,17.4l1.4,1.4 c0.4,0.4,1,0.4,1.4,0c0.4-0.4,0.4-1,0-1.4l-1.4-1.4c-0.4-0.4-1-0.4-1.4,0C17.2,16.2,17.2,16.8,18.4,17.4z M6.4,19.4L5,18 c-0.4-0.4-1-0.4-1.4,0c-0.4,0.4-0.4,1,0,1.4l1.4,1.4c0.4,0.4,1,0.4,1.4,0C6.8,20.4,6.8,19.8,6.4,19.4z M18.4,4.6l-1.4-1.4 c-0.4-0.4-1-0.4-1.4,0c-0.4,0.4-0.4,1,0,1.4l1.4,1.4c0.4,0.4,1,0.4,1.4,0C18.8,5.6,18.8,5,18.4,4.6z"
            />
          </svg>
        </button>

        <button
          className={clsx(
            styles.themeButton,
            theme === 'auto' && styles.themeButtonActive
          )}
          onClick={setAutoTheme}
          aria-label="Switch to automatic theme"
          title="Auto theme"
        >
          <svg
            className={styles.themeIcon}
            viewBox="0 0 24 24"
            width="18"
            height="18"
          >
            <path
              fill="currentColor"
              d="M12,1L12,1C6.5,1,2,5.5,2,11v0c0,5.5,4.5,10,10,10v0c5.5,0,10-4.5,10-10v0C22,5.5,17.5,1,12,1z M12,19c-4.4,0-8-3.6-8-8 s3.6-8,8-8s8,3.6,8,8S16.4,19,12,19z M13,6h-2v2h2V6 M12,11c-2.2,0-4,1.8-4,4h2c0-1.1,0.9-2,2-2s2,0.9,2,2c1.1,0,2-0.9,2-2 C16,12.8,14.2,11,12,11z"
            />
          </svg>
        </button>

        <button
          className={clsx(
            styles.themeButton,
            theme === 'dark' && styles.themeButtonActive
          )}
          onClick={() => {
            setTheme('dark');
            applyTheme('dark', true);
          }}
          aria-label="Switch to dark theme"
          title="Dark theme"
        >
          <svg
            className={styles.themeIcon}
            viewBox="0 0 24 24"
            width="18"
            height="18"
          >
            <path
              fill="currentColor"
              d="M9.37,5.39C8,6.23,7,7.9,7,9.5C7,10.8,7.47,12,8.24,13H7v2h6v-1.76c1.2,0.77,2,2,2,3.5c0,2.21-1.79,4-4,4 s-4-1.79-4-4c0-0.7,0.21-1.35,0.59-1.91C5.73,14.16,5,12.23,5,10C5,6.69,7.69,4,11,4C12.23,4,13.38,4.35,14.37,5H13V7H11V5H9.37z M11,6 c-1.66,0-3,1.34-3,3c0,1.66,1.34,3,3,3s3-1.34,3-3C14,7.34,12.66,6,11,6L11,6z"
            />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default ThemeToggle;