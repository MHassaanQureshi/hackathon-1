import React from 'react';
import Layout from '@theme/Layout';
import AIBotDocumentation from '../components/AIBotDocumentation';

function AIBotPage() {
  return (
    <Layout title="AI Assistant" description="Interactive AI assistant for the AI-Native Book">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--12">
            <AIBotDocumentation />
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default AIBotPage;