import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/intro">
            Start Learning
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/module-1-the-robotic-nervous-system">
            Explore Modules
          </Link>
        </div>
      </div>
    </header>
  );
}

function ModulesGrid() {
  return (
    <section className={styles.modulesSection}>
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <div className={clsx('card', styles.moduleCard)}>
              <div className="card__header">
                <h3>Module 1: Robotic Nervous System (ROS 2)</h3>
              </div>
              <div className="card__body">
                <p>Learn ROS 2 fundamentals, nodes, topics, services, and Python agents with rclpy.</p>
              </div>
              <div className="card__footer">
                <Link className="button button--primary" to="/docs/module-1-the-robotic-nervous-system">
                  Start Module
                </Link>
              </div>
            </div>
          </div>
          <div className="col col--6">
            <div className={clsx('card', styles.moduleCard)}>
              <div className="card__header">
                <h3>Module 2: Digital Twin (Gazebo & Unity)</h3>
              </div>
              <div className="card__body">
                <p>Physics simulation in Gazebo, high-fidelity environments in Unity, and sensor simulation.</p>
              </div>
              <div className="card__footer">
                <Link className="button button--primary" to="/docs/module-2-the-digital-twin">
                  Start Module
                </Link>
              </div>
            </div>
          </div>
        </div>
        <div className="row" style={{marginTop: '2rem'}}>
          <div className="col col--6">
            <div className={clsx('card', styles.moduleCard)}>
              <div className="card__header">
                <h3>Module 3: AI-Robot Brain (NVIDIA Isaac)</h3>
              </div>
              <div className="card__body">
                <p>Isaac Sim and synthetic data, Isaac ROS and VSLAM, Navigation and path planning with Nav2.</p>
              </div>
              <div className="card__footer">
                <Link className="button button--primary" to="/docs/module-3-the-ai-robot-brain">
                  Start Module
                </Link>
              </div>
            </div>
          </div>
          <div className="col col--6">
            <div className={clsx('card', styles.moduleCard)}>
              <div className="card__header">
                <h3>Module 4: Vision-Language-Action (VLA)</h3>
              </div>
              <div className="card__body">
                <p>Voice commands with OpenAI Whisper, LLM-based task planning, and autonomous humanoid robot capstone.</p>
              </div>
              <div className="card__footer">
                <Link className="button button--primary" to="/docs/module-4-vision-language-action">
                  Start Module
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`AI-Native Book - Physical AI & Humanoid Robotics`}
      description="A comprehensive guide to Physical AI systems and humanoid robot control">
      <HomepageHeader />
      <main>
        <ModulesGrid />
      </main>
    </Layout>
  );
}
