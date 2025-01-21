import React from 'react';
import { Button } from "./button";
import { Input } from "./input";
import { Link } from "lucide-react";

// Update these paths according to your project structure
import youtubeIcon from '../../assets/icons/youtube_social_squircle_red.png';
import spotifyIcon from '../../assets/icons/spotify-icon.png';

export const LinkSection = ({ 
  linkUrl, 
  setLinkUrl, 
  isProcessingLink, 
  processLink 
}) => {
  return (
    <div className="space-y-6">
      <div className="relative flex items-center gap-2">
        <Input
          type="url"
          placeholder="Paste YouTube, Spotify, or audio URL"
          value={linkUrl}
          onChange={(e) => setLinkUrl(e.target.value)}
          className="w-full h-12 px-4 text-base border-2 border-fuchsia-200 focus:border-fuchsia-400 focus:ring-fuchsia-400"
        />
        <Button 
          size="lg"
          className="h-12 px-6 bg-fuchsia-600 hover:bg-fuchsia-700 text-white font-medium whitespace-nowrap flex items-center justify-center"
          onClick={async () => {
            if (!linkUrl.trim()) return;
            await processLink(linkUrl);
          }}
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
                src={spotifyIcon} 
                alt="Spotify" 
                className="w-5 h-5"
              />
              <span className="font-medium">Spotify</span>
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
        {/* <p className="text-sm text-fuchsia-500/80 italic">
          Note: Some audio sources may provide preview clips only
        </p> */}
      </div>
    </div>
  );
};