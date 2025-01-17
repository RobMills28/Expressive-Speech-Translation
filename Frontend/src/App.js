import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Card } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Globe } from 'lucide-react';
import TranslateTool from './components/TranslateTool';
import Dashboard from "./components/Dashboard";

// Navigation Component
const Navigation = () => {
  const location = useLocation();

  // Don't show navigation on landing page
  if (location.pathname === '/') {
    return null;
  }
  
  return (
    <nav className="bg-white border-b">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="text-xl font-bold text-fuchsia-600">
              Magenta AI
            </Link>
          </div>
          
          <div className="flex items-center gap-4">
            <Link 
              to="/translate"
              className={`px-4 py-2 text-gray-600 hover:text-gray-900 ${
                location.pathname === '/translate' ? 'text-fuchsia-600' : ''
              }`}
            >
              Translation Tool
            </Link>
            <Link 
              to="/dashboard"
              className={`px-4 py-2 rounded-lg hover:bg-fuchsia-700 ${
                location.pathname === '/dashboard' 
                  ? 'bg-fuchsia-700 text-white'
                  : 'bg-fuchsia-600 text-white'
              }`}
            >
              Dashboard
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

// Landing Page Component
const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-fuchsia-50 to-white">
      <nav className="border-b bg-white">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-fuchsia-600">Magenta AI</h1>
          <div className="flex items-center gap-4">
            <Link
              to="/translate"
              className="px-4 py-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              Translation Tool
            </Link>
            <Link
              to="/dashboard"
              className="px-4 py-2 bg-fuchsia-600 text-white rounded-lg hover:bg-fuchsia-700 transition-colors"
            >
              Dashboard
            </Link>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto pt-32">
          <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-fuchsia-600 to-pink-600">
            Share Your Content With the World, In Any Language
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Magenta helps creators reach global audiences by translating content while 
            preserving their authentic voice and style.
          </p>
          <Link
            to="/translate"
            className="inline-block px-8 py-4 bg-fuchsia-600 text-white rounded-lg hover:bg-fuchsia-700 text-lg shadow-lg hover:shadow-xl transition-all duration-200"
          >
            Get Started Free
          </Link>
        </div>

        <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute -top-1/2 left-1/2 -translate-x-1/2 w-full max-w-4xl h-[800px] rounded-full bg-fuchsia-100/50 blur-3xl" />
          <div className="absolute top-1/4 right-1/4 w-[600px] h-[600px] rounded-full bg-pink-100/30 blur-3xl" />
        </div>

        <div className="max-w-6xl mx-auto mt-16 px-4">
          <div className="aspect-video w-full max-w-3xl mx-auto rounded-xl bg-white/50 shadow-lg backdrop-blur-sm border border-white/20" />
        </div>
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-white">
        <Navigation />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/translate" element={<TranslateTool />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;