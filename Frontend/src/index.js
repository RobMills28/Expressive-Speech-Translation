import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import { AuthProvider } from "react-oidc-context";

const cognitoAuthConfig = {
  authority: "https://cognito-idp.eu-west-2.amazonaws.com/eu-west-2_u9DeSOC1s",
  client_id: "6pr8qneun0lhjorncf8bgi61t2",  // Use your actual client ID
  redirect_uri: "https://magentaplatform.com/dashboard",
  response_type: "code",
  scope: "phone openid email"
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <AuthProvider {...cognitoAuthConfig}>
      <App />
    </AuthProvider>
  </React.StrictMode>
);