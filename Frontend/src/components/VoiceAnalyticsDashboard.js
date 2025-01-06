// src/components/VoiceAnalyticsDashboard.js
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { LineChart, Line, BarChart, Bar, RadarChart, Radar, PolarGrid, 
         PolarAngleAxis, ResponsiveContainer, XAxis, YAxis, Tooltip, 
         Legend, PolarRadiusAxis } from 'recharts';
import { Volume2, Mic2, Timer, Activity, WaveformIcon } from 'lucide-react';

const VoiceAnalyticsDashboard = () => {
  // Sample data for visualizations
  const timeSeriesData = [
    { time: '0:00', sourcePitch: 220, targetPitch: 230 },
    { time: '0:30', sourcePitch: 235, targetPitch: 240 },
    { time: '1:00', sourcePitch: 210, targetPitch: 215 },
    { time: '1:30', sourcePitch: 225, targetPitch: 228 },
    { time: '2:00', sourcePitch: 240, targetPitch: 235 },
    { time: '2:30', sourcePitch: 230, targetPitch: 232 },
  ];

  const emotionData = [
    { name: 'Neutral', source: 65, target: 68 },
    { name: 'Happy', source: 45, target: 42 },
    { name: 'Serious', source: 30, target: 35 },
    { name: 'Energetic', source: 25, target: 28 },
    { name: 'Calm', source: 20, target: 22 },
  ];

  const radarData = [
    { category: 'Volume', source: 85, target: 82 },
    { category: 'Pace', source: 78, target: 75 },
    { category: 'Pitch', source: 90, target: 88 },
    { category: 'Clarity', source: 85, target: 87 },
    { category: 'Emotion', source: 75, target: 78 },
  ];

  return (
    <div className="container mx-auto p-6">
      {/* Header */}
      <div className="text-white mb-6">
        <h1 className="text-3xl font-bold">Voice Analytics</h1>
        <p className="text-gray-200">Detailed analysis of voice translation metrics</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {[
          { title: 'Average Volume', value: '72 dB', icon: Volume2, change: '+2.4%' },
          { title: 'Speech Rate', value: '145 WPM', icon: Timer, change: '-1.2%' },
          { title: 'Voice Clarity', value: '94%', icon: Mic2, change: '+3.7%' },
          { title: 'Emotion Match', value: '88%', icon: Activity, change: '+1.5%' },
        ].map((stat, index) => (
          <Card key={index} className="bg-white/90 backdrop-blur-md">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                  <h3 className="text-2xl font-bold text-gray-900">{stat.value}</h3>
                </div>
                <div className={`p-2 rounded-full ${
                  stat.change.startsWith('+') ? 'bg-green-100' : 'bg-red-100'
                }`}>
                  <stat.icon className={`w-5 h-5 ${
                    stat.change.startsWith('+') ? 'text-green-600' : 'text-red-600'
                  }`} />
                </div>
              </div>
              <p className={`mt-2 text-sm ${
                stat.change.startsWith('+') ? 'text-green-600' : 'text-red-600'
              }`}>
                {stat.change} from source
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pitch Analysis */}
        <Card className="bg-white/90 backdrop-blur-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Pitch Analysis Over Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeSeriesData}>
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="sourcePitch" 
                    stroke="#8b5cf6" 
                    name="Source Voice"
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="targetPitch" 
                    stroke="#ec4899" 
                    name="Translated Voice"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Voice Quality Comparison */}
        <Card className="bg-white/90 backdrop-blur-md">
          <CardHeader>
            <CardTitle>Voice Quality Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="category" />
                  <PolarRadiusAxis />
                  <Radar 
                    name="Source" 
                    dataKey="source" 
                    stroke="#8b5cf6" 
                    fill="#8b5cf6" 
                    fillOpacity={0.5} 
                  />
                  <Radar 
                    name="Target" 
                    dataKey="target" 
                    stroke="#ec4899" 
                    fill="#ec4899" 
                    fillOpacity={0.5} 
                  />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Emotion Distribution */}
        <Card className="bg-white/90 backdrop-blur-md">
          <CardHeader>
            <CardTitle>Emotion Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={emotionData}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="source" name="Source Voice" fill="#8b5cf6" />
                  <Bar dataKey="target" name="Translated Voice" fill="#ec4899" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default VoiceAnalyticsDashboard;