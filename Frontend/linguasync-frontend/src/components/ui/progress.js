import React from 'react';

export function Progress({ value, className, ...props }) {
  return (
    <div className={`w-full bg-gray-200 rounded ${className}`} {...props}>
      <div 
        className="bg-blue-600 text-xs font-medium text-blue-100 text-center p-0.5 leading-none rounded" 
        style={{ width: `${value}%` }}
      >
        {value}%
      </div>
    </div>
  );
}