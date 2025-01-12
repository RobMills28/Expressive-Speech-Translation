import React from 'react';
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Home, Globe, PlayCircle, BarChart2, Users, Settings } from 'lucide-react';

const Dashboard = () => {
  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <div className="w-64 border-r bg-white">
        <div className="px-6 py-4">
          <h1 className="text-xl font-bold text-purple-600">Magenta</h1>
        </div>

        <div className="px-4">
          <Button className="w-full mb-8 bg-purple-600 hover:bg-purple-700">
            ↑ New Upload
          </Button>

          <div className="space-y-8">
            <div>
              <p className="text-xs font-medium text-gray-500 mb-3">WORKSPACE</p>
              <div className="space-y-1">
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-2 py-2">
                  <Home className="h-4 w-4" />
                  <span className="ml-3 text-sm">Overview</span>
                </div>
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-2 py-2">
                  <Globe className="h-4 w-4" />
                  <span className="ml-3 text-sm">Translations</span>
                  <span className="ml-auto bg-purple-100 text-purple-600 px-2 rounded-full text-xs">3</span>
                </div>
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-2 py-2">
                  <PlayCircle className="h-4 w-4" />
                  <span className="ml-3 text-sm">Content</span>
                </div>
              </div>
            </div>

            <div>
              <p className="text-xs font-medium text-gray-500 mb-3">DISTRIBUTION</p>
              <div className="space-y-1">
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-2 py-2">
                  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z" />
                  </svg>
                  <span className="ml-3 text-sm">YouTube</span>
                </div>
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-2 py-2">
                  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z" />
                  </svg>
                  <span className="ml-3 text-sm">Spotify</span>
                </div>
              </div>
            </div>

            <div>
              <p className="text-xs font-medium text-gray-500 mb-3">INSIGHTS</p>
              <div className="space-y-1">
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-2 py-2">
                  <BarChart2 className="h-4 w-4" />
                  <span className="ml-3 text-sm">Analytics</span>
                </div>
                <div className="flex items-center text-gray-800 hover:bg-gray-100 rounded-lg px-2 py-2">
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
            <div>
              <h2 className="text-2xl font-semibold mb-1">Creator Studio</h2>
              <p className="text-gray-500">Manage your content and translations</p>
            </div>
            <div className="flex gap-4">
              <Input 
                placeholder="Search content..." 
                className="w-64"
              />
              <Button className="bg-purple-600 hover:bg-purple-700">New Project</Button>
            </div>
          </div>

          {/* Recent Activity */}
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

          {/* Translation Progress */}
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
                    <div className="h-2 bg-purple-600 rounded-full" style={{ width: '65%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="font-medium">Spanish</span>
                    <span className="text-gray-500">100%</span>
                  </div>
                  <div className="h-2 bg-gray-100 rounded-full">
                    <div className="h-2 bg-purple-600 rounded-full" style={{ width: '100%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="font-medium">German</span>
                    <span className="text-gray-500">35%</span>
                  </div>
                  <div className="h-2 bg-gray-100 rounded-full">
                    <div className="h-2 bg-purple-600 rounded-full" style={{ width: '35%' }}></div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;