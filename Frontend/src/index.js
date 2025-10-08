// src/index.js

import React from 'react';
import ReactDOM from 'react-dom/client';

import '@aws-amplify/ui-react/styles.css'; // 1. Import default styles FIRST
import './index.css';                     // 2. Import your theme overrides SECOND

import App from './App';
import { Amplify } from 'aws-amplify';

Amplify.configure({ /* ... your config object ... */ });

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);