import React from 'react';

export function Alert({ children, className, ...props }) {
  return <div className={`p-4 border rounded ${className}`} {...props}>{children}</div>;
}

export function AlertTitle({ children, className, ...props }) {
  return <h5 className={`font-medium ${className}`} {...props}>{children}</h5>;
}

export function AlertDescription({ children, className, ...props }) {
  return <div className={`text-sm ${className}`} {...props}>{children}</div>;
}
