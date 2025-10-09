// src/index.js

import React from 'react';
import ReactDOM from 'react-dom/client';

// These style imports are correct. Do not change them.
import '@aws-amplify/ui-react/styles.css';
import './index.css';

import App from './App';
import { Amplify } from 'aws-amplify';

// --- START: THIS IS THE FIX ---
// We are replacing the placeholder with your actual Cognito details.
Amplify.configure({
  Auth: {
    Cognito: {
      userPoolId: 'us-east-1_idzaATOBT',
      userPoolClientId: 'jtbqug4a9qe04sgp0l7235g6h',
      loginWith: {
        oauth: {
          domain: 'us-east-1dzaatobt.auth.us-east-1.amazoncognito.com',
          scopes: ['openid', 'email', 'profile'],
          redirectSignIn: [
            'http://localhost:3000/creator-studio',
            'https://www.contentconverter.ai/creator-studio' // <-- Added Production URL
          ],
          redirectSignOut: [
            'http://localhost:3000/',
            'https://www.contentconverter.ai/' // <-- Added Production URL
          ],
          responseType: 'code',
        }
      }
    }
  }
});
// --- END: THIS IS THE FIX ---

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);