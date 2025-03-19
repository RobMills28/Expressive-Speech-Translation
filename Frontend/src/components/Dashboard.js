import React, { useState } from 'react';
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Upload, Home, Globe, PlayCircle, BarChart2, Users, Film, Folder } from 'lucide-react';
import TranslationFlow from './TranslationFlow';
import VideoSyncInterface from './VideoSyncInterface';

import youtubeIcon from '../assets/icons/youtube_social_squircle_red.png';
import spotifyIcon from '../assets/icons/spotify-icon.png';
import tiktokIcon from '../assets/icons/TikTok_Icon_Black_Circle.png';

const Dashboard = () => {
  const [showTranslationFlow, setShowTranslationFlow] = useState(false);
  const [activeView, setActiveView] = useState('newTranslation');

  const renderMainContent = () => {
    switch(activeView) {
      case 'videoSync':
        return <VideoSyncInterface />;
      case 'dashboard':
        return (
          <>
            <Card className="mb-8">
              <CardContent className="p-8">
                <h3 className="text-lg font-semibold mb-6">Recent Activity</h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center bg-gray-50 rounded-lg border px-6 py-4">
                    <div>
                      <h4 className="font-medium">The Future of AI</h4>
                      <p className="text-sm text-gray-500">Translation completed</p>
                    </div>
                    <span className="text-sm text-gray-500">2h ago</span>
                  </div>
                  <div className="flex justify-between items-center bg-gray-50 rounded-lg border px-6 py-4">
                    <div>
                      <h4 className="font-medium">Web3 and Digital Finance</h4>
                      <p className="text-sm text-gray-500">Processing translations</p>
                    </div>
                    <span className="text-sm text-gray-500">5h ago</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-8">
                <h3 className="text-lg font-semibold mb-6">Translation Progress</h3>
                <div className="space-y-6">
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
          </>
        );
      case 'library':
        return (
          <Card>
            <CardContent className="p-8 text-center">
              <Folder className="w-16 h-16 mx-auto text-gray-300 mb-4" />
              <h3 className="text-lg font-semibold mb-2">Your translations will appear here</h3>
              <p className="text-gray-500 mb-6">Once you create translations, they'll be saved to your library</p>
              <Button 
                className="bg-fuchsia-600 hover:bg-fuchsia-700"
                onClick={() => setActiveView('newTranslation')}
              >
                Create Your First Translation
              </Button>
            </CardContent>
          </Card>
        );
      case 'newTranslation':
      default:
        return (
          <div className="max-w-4xl mx-auto">
            <TranslationFlow />
          </div>
        );
    }
  };

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <div className="w-64 border-r bg-white">
        <div className="px-6 py-4">
          <h1 className="text-xl font-bold text-fuchsia-600 mb-8">Magenta AI</h1>

          <div className="space-y-8">
            <div>
              <p className="text-xs font-medium text-gray-500 mb-3">WORKSPACE</p>
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
                <div 
                  className={`flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-3 py-2.5 cursor-pointer ${
                    activeView === 'videoSync' ? 'bg-fuchsia-50 text-fuchsia-700' : ''
                  }`}
                  onClick={() => setActiveView('videoSync')}
                >
                  <Film className={`h-4 w-4 ${activeView === 'videoSync' ? 'text-fuchsia-600' : ''}`} />
                  <span className="ml-3 text-sm">Video Sync</span>
                </div>
              </div>
            </div>

            <div>
              <p className="text-xs font-medium text-gray-500 mb-3">DISTRIBUTION</p>
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
              <p className="text-xs font-medium text-gray-500 mb-3">INSIGHTS</p>
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
          <div className="flex justify-between items-center mb-8">
            <div className="flex-1">
              {activeView === 'newTranslation' && (
                <>
                  <h2 className="text-2xl font-semibold mb-1">New Translation</h2>
                  <p className="text-gray-500">Upload and translate your content</p>
                </>
              )}
              {activeView === 'dashboard' && (
                <>
                  <h2 className="text-2xl font-semibold mb-1">Dashboard</h2>
                  <p className="text-gray-500">Overview of your activities and progress</p>
                </>
              )}
              {activeView === 'library' && (
                <>
                  <h2 className="text-2xl font-semibold mb-1">Library</h2>
                  <p className="text-gray-500">Browse all your translation projects</p>
                </>
              )}
              {activeView === 'videoSync' && (
                <>
                  <h2 className="text-2xl font-semibold mb-1">Video Sync</h2>
                  <p className="text-gray-500">Sync your video content with translations</p>
                </>
              )}
            </div>
            <div className="flex gap-4 items-center">
              {activeView !== 'newTranslation' && (
                <Button 
                  className="bg-fuchsia-600 hover:bg-fuchsia-700 whitespace-nowrap"
                  onClick={() => setActiveView('newTranslation')}
                >
                  New Project
                </Button>
              )}
            </div>
          </div>

          {renderMainContent()}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;