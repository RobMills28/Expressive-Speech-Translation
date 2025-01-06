// src/components/Navigation.js
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Mic, Radio, BarChart2 } from 'lucide-react';

const Navigation = () => {
  const location = useLocation();
  
  return (
    <div className="max-w-7xl mx-auto px-4">
      <div className="flex items-center justify-between h-16">
        <div className="flex items-center">
          <Link to="/" className="text-2xl font-bold text-purple-400 hover:text-purple-300 transition-colors">
            LinguaSync
          </Link>
        </div>
        
        <div className="flex items-center space-x-4">
          <Link 
            to="/"
            className={`flex items-center px-3 py-2 rounded-md text-sm font-medium ${
              location.pathname === '/' 
                ? 'bg-purple-600 text-white' 
                : 'text-gray-300 hover:bg-gray-700 hover:text-white'
            }`}
          >
            <Mic className="w-4 h-4 mr-2" />
            Translate
          </Link>
          
          <Link 
            to="/podcasts"
            className={`flex items-center px-3 py-2 rounded-md text-sm font-medium ${
              location.pathname === '/podcasts' 
                ? 'bg-purple-600 text-white' 
                : 'text-gray-300 hover:bg-gray-700 hover:text-white'
            }`}
          >
            <Radio className="w-4 h-4 mr-2" />
            Podcasts
          </Link>

          <Link 
            to="/analytics"
            className={`flex items-center px-3 py-2 rounded-md text-sm font-medium ${
              location.pathname === '/analytics' 
                ? 'bg-purple-600 text-white' 
                : 'text-gray-300 hover:bg-gray-700 hover:text-white'
            }`}
          >
            <BarChart2 className="w-4 h-4 mr-2" />
            Analytics
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Navigation;