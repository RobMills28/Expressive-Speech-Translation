import React, { useState } from 'react';
import { Button } from "./button";
import { Input } from "./input";
import { Link, AlertCircle } from "lucide-react";
import { Alert, AlertDescription } from "./alert";

// Update these paths according to your project structure
import youtubeIcon from '../../assets/icons/youtube_social_squircle_red.png';
import tiktokIcon from '../../assets/icons/TikTok_Icon_Black_Circle.png';

export const LinkSection = ({ 
  linkUrl, 
  setLinkUrl, 
  isProcessingLink, 
  processLink 
}) => {
  const [showSpotifyMessage, setShowSpotifyMessage] = useState(false);
  
  const handleProcessLink = async () => {
    if (!linkUrl.trim()) return;
    
    if (linkUrl.includes('spotify.com')) {
      setShowSpotifyMessage(true);
      return;
    }
    
    await processLink(linkUrl);
    setShowSpotifyMessage(false);
  };

  return (
    <div className="space-y-6">
      <div className="relative flex items-center gap-2">
        <Input
          type="url"
          placeholder="Paste YouTube, TikTok, or audio URL"
          value={linkUrl}
          onChange={(e) => {
            setLinkUrl(e.target.value);
            setShowSpotifyMessage(false);
          }}
          className="w-full h-12 px-4 text-base border-2 border-fuchsia-200 focus:border-fuchsia-400 focus:ring-fuchsia-400"
        />
        <Button 
          size="lg"
          className="h-12 px-6 bg-fuchsia-600 hover:bg-fuchsia-700 text-white font-medium whitespace-nowrap flex items-center justify-center"
          onClick={handleProcessLink}
          disabled={!linkUrl.trim() || isProcessingLink}
        >
          {isProcessingLink ? (
            <div className="flex items-center gap-2">
              <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
              <span>Processing</span>
            </div>
          ) : (
            'Process Link'
          )}
        </Button>
      </div>

      {showSpotifyMessage && (
        <Alert className="bg-amber-50 border-amber-200">
          <AlertCircle className="h-4 w-4 text-amber-600" />
          <AlertDescription className="text-amber-800">
            Spotify tracks aren't currently supported. Try YouTube or TikTok instead!
          </AlertDescription>
        </Alert>
      )}

      <div className="space-y-3">
        <div className="text-sm text-fuchsia-600">
          <span className="mr-2">Supported sources:</span>
          <div className="flex flex-wrap gap-2 mt-2">
            <div className="px-3 py-1.5 bg-white rounded-full border border-fuchsia-200 shadow-sm flex items-center gap-2">
              <img 
                src={youtubeIcon} 
                alt="YouTube" 
                className="w-5 h-5"
              />
              <span className="font-medium">YouTube</span>
            </div>
            <div className="px-3 py-1.5 bg-white rounded-full border border-fuchsia-200 shadow-sm flex items-center gap-2">
              <img 
                src={tiktokIcon} 
                alt="TikTok" 
                className="w-5 h-5"
              />
              <span className="font-medium">TikTok</span>
            </div>
            <div className="px-3 py-1.5 bg-white rounded-full border border-fuchsia-200 shadow-sm flex items-center gap-2">
              <Link size={16} className="text-fuchsia-600" />
              <span className="font-medium">Direct audio</span>
            </div>
          </div>
        </div>
        <p className="text-sm text-fuchsia-500/80">
          Some audio sources may provide preview clips only
        </p>
      </div>
    </div>
  );
};