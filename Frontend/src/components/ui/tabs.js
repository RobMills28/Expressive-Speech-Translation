import React from 'react';

export const Tabs = ({ defaultValue, children, className }) => {
  const [activeTab, setActiveTab] = React.useState(defaultValue);
  
  return (
    <div className={className}>
      {React.Children.map(children, child => {
        if (child.type === TabsList || child.type === TabsContent) {
          return React.cloneElement(child, { activeTab, setActiveTab });
        }
        return child;
      })}
    </div>
  );
};

export const TabsList = ({ children, activeTab, setActiveTab, className }) => {
  return (
    <div className={className}>
      {React.Children.map(children, child => {
        if (child.type === TabsTrigger) {
          return React.cloneElement(child, { 
            isActive: activeTab === child.props.value,
            onClick: () => setActiveTab(child.props.value),
          });
        }
        return child;
      })}
    </div>
  );
};

export const TabsTrigger = ({ value, isActive, onClick, children, className }) => {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`${className} ${
        isActive ? 'bg-white text-fuchsia-700 shadow-sm' : 'text-gray-500 hover:text-gray-700'
      }`}
    >
      {children}
    </button>
  );
};

export const TabsContent = ({ value, activeTab, children, className }) => {
  if (value !== activeTab) return null;
  
  return (
    <div className={className}>
      {children}
    </div>
  );
};