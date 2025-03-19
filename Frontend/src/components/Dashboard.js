import React, { useState } from 'react';
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Upload, Home, Globe, BarChart2, Users, Folder } from 'lucide-react';
import TranslationFlow from './TranslationFlow';

import youtubeIcon from '../assets/icons/youtube_social_squircle_red.png';
import spotifyIcon from '../assets/icons/spotify-icon.png';
import tiktokIcon from '../assets/icons/TikTok_Icon_Black_Circle.png';

const Dashboard = () => {
  const [activeView, setActiveView] = useState('newTranslation');

  // Header with "Creator Studio" button
  const renderHeader = () => (
    <div className="flex justify-between items-center mb-6 p-3 bg-white rounded-lg shadow-sm">
      <h1 className="text-xl font-bold text-fuchsia-600">Magenta AI</h1>
      <Button 
        className="bg-fuchsia-600 hover:bg-fuchsia-700"
        onClick={() => {}}
      >
        Creator Studio
      </Button>
    </div>
  );

  const renderMainContent = () => {
    switch(activeView) {
      case 'dashboard':
        return (
          <div>
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-2xl font-medium mb-1">Dashboard</h2>
                <p className="text-gray-500">Overview of your activities and progress</p>
              </div>
              <Button 
                className="bg-fuchsia-600 hover:bg-fuchsia-700"
                onClick={() => setActiveView('newTranslation')}
              >
                New Project
              </Button>
            </div>
            <div className="space-y-6">
              <Card className="shadow-sm">
                <CardContent className="p-6">
                  <h3 className="text-lg font-medium mb-4">Recent Activity</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center bg-gray-50 rounded-md border px-4 py-3">
                      <div>
                        <h4 className="font-medium">The Future of AI</h4>
                        <p className="text-sm text-gray-500">Translation completed</p>
                      </div>
                      <span className="text-sm text-gray-500">2h ago</span>
                    </div>
                    <div className="flex justify-between items-center bg-gray-50 rounded-md border px-4 py-3">
                      <div>
                        <h4 className="font-medium">Web3 and Digital Finance</h4>
                        <p className="text-sm text-gray-500">Processing translations</p>
                      </div>
                      <span className="text-sm text-gray-500">5h ago</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-sm">
                <CardContent className="p-6">
                  <h3 className="text-lg font-medium mb-4">Translation Progress</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="font-medium">French</span>
                        <span className="text-gray-500">65%</span>
                      </div>
                      <div className="h-2 bg-gray-100 rounded-full">
                        <div className="h-2 bg-fuchsia-600 rounded-full" style={{ width: '65%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="font-medium">Spanish</span>
                        <span className="text-gray-500">100%</span>
                      </div>
                      <div className="h-2 bg-gray-100 rounded-full">
                        <div className="h-2 bg-fuchsia-600 rounded-full" style={{ width: '100%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="font-medium">German</span>
                        <span className="text-gray-500">35%</span>
                      </div>
                      <div className="h-2 bg-gray-100 rounded-full">
                        <div className="h-2 bg-fuchsia-600 rounded-full" style={{ width: '35%' }}></div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        );
      case 'library':
        return (
          <div>
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-2xl font-medium mb-1">Library</h2>
                <p className="text-gray-500">Browse all your translation projects</p>
              </div>
              <Button 
                className="bg-fuchsia-600 hover:bg-fuchsia-700"
                onClick={() => setActiveView('newTranslation')}
              >
                New Project
              </Button>
            </div>
            <Card className="shadow-sm">
              <CardContent className="p-12 text-center">
                <Folder className="w-16 h-16 mx-auto text-gray-300 mb-4" />
                <h3 className="text-lg font-medium mb-2">Your translations will appear here</h3>
                <p className="text-gray-500 mb-6 max-w-md mx-auto">Once you create translations, they'll be saved to your library for easy access</p>
                <Button 
                  className="bg-fuchsia-600 hover:bg-fuchsia-700"
                  onClick={() => setActiveView('newTranslation')}
                >
                  Create Your First Translation
                </Button>
              </CardContent>
            </Card>
          </div>
        );
      case 'newTranslation':
      default:
        return (
          <div>
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-2xl font-medium mb-1">New Translation</h2>
                <p className="text-gray-500">Upload and translate your content</p>
              </div>
            </div>
            <TranslationFlow />
          </div>
        );
    }
  };

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <div className="w-64 border-r bg-white">
        <div className="p-4">
          <div className="mb-6">
            <h1 className="text-xl font-bold text-fuchsia-600">Magenta AI</h1>
          </div>

          <div className="space-y-6">
            <div>
              <p className="text-xs font-medium text-gray-500 mb-2">WORKSPACE</p>
              <div className="space-y-1">
                <div 
                  className={`flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-3 py-2.5 cursor-pointer ${
                    activeView === 'newTranslation' ? 'bg-fuchsia-50 text-fuchsia-700' : ''
                  }`}
                  onClick={() => setActiveView('newTranslation')}
                >
                  <Upload className={`h-4 w-4 ${activeView === 'newTranslation' ? 'text-fuchsia-600' : ''}`} />
                  <span className="ml-3 text-sm">New Translation</span>
                </div>
                <div 
                  className={`flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-3 py-2.5 cursor-pointer ${
                    activeView === 'dashboard' ? 'bg-fuchsia-50 text-fuchsia-700' : ''
                  }`}
                  onClick={() => setActiveView('dashboard')}
                >
                  <Home className={`h-4 w-4 ${activeView === 'dashboard' ? 'text-fuchsia-600' : ''}`} />
                  <span className="ml-3 text-sm">Dashboard</span>
                </div>
                <div 
                  className={`flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-3 py-2.5 cursor-pointer ${
                    activeView === 'library' ? 'bg-fuchsia-50 text-fuchsia-700' : ''
                  }`}
                  onClick={() => setActiveView('library')}
                >
                  <Folder className={`h-4 w-4 ${activeView === 'library' ? 'text-fuchsia-600' : ''}`} />
                  <span className="ml-3 text-sm">Library</span>
                </div>
              </div>
            </div>

            <div>
              <p className="text-xs font-medium text-gray-500 mb-2">DISTRIBUTION</p>
              <div className="space-y-1">
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-3 py-2.5">
                  <img 
                    src={youtubeIcon} 
                    alt="YouTube" 
                    className="w-5 h-5"
                  />
                  <span className="ml-3 text-sm">YouTube</span>
                </div>
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-3 py-2.5">
                  <img 
                    src={spotifyIcon} 
                    alt="Spotify" 
                    className="w-5 h-5"
                  />
                  <span className="ml-3 text-sm">Spotify</span>
                </div>
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-3 py-2.5">
                  <img 
                    src={tiktokIcon} 
                    alt="TikTok" 
                    className="w-5 h-5"
                  />
                  <span className="ml-3 text-sm">TikTok</span>
                </div>
              </div>
            </div>

            <div>
              <p className="text-xs font-medium text-gray-500 mb-2">INSIGHTS</p>
              <div className="space-y-1">
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-3 py-2.5">
                  <BarChart2 className="h-4 w-4" />
                  <span className="ml-3 text-sm">Analytics</span>
                </div>
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-3 py-2.5">
                  <Users className="h-4 w-4" />
                  <span className="ml-3 text-sm">Audience</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto bg-gray-50">
        <div className="p-8">
          {renderMainContent()}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;