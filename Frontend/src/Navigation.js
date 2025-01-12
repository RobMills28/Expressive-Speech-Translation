import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom'; // Added useNavigate
import { Mic, Radio, BarChart2 } from 'lucide-react';

const Navigation = () => {
  const location = useLocation();
  const navigate = useNavigate();  // Add this
  
  return (
    <div className="max-w-7xl mx-auto px-4">
      <div className="flex items-center justify-between h-16">
        <div className="flex items-center">
          <Link to="/" className="text-2xl font-bold text-purple-400 hover:text-purple-300 transition-colors">
            LinguaSync
          </Link>
          {/* Add test button here */}
          <button 
            onClick={() => navigate('/temp_dashboard')}
            className="ml-4 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
          >
            Test Dashboard
          </button>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Rest of your navigation links */}
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
            to="/temp_dashboard"
            className={`flex items-center px-3 py-2 rounded-md text-sm font-medium ${
              location.pathname === '/temp_dashboard' 
              ? 'bg-purple-600 text-white' 
              : 'text-gray-300 hover:bg-gray-700 hover:text-white'
            }`}
          >
            <Radio className="w-4 h-4 mr-2" />
            Dashboard
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