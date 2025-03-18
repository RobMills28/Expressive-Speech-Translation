import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Check, HelpCircle } from 'lucide-react';
import { Button } from "./ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "./ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";

const FeatureItem = ({ children, included = true, tooltip }) => {
    if (tooltip) {
      return (
        <TooltipProvider>
          <div className="flex items-center">
            {included ? (
              <Check className="h-4 w-4 text-fuchsia-500 mr-2" />
            ) : (
              <div className="h-4 w-4 mr-2" />
            )}
            <span className={included ? "" : "text-gray-400"}>{children}</span>
            <Tooltip>
              <TooltipTrigger>
                <HelpCircle className="h-4 w-4 text-gray-400 ml-2 cursor-help" />
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-xs">
                <p className="text-sm">{tooltip}</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </TooltipProvider>
      );
    }
  
    return (
      <div className="flex items-center">
        {included ? (
          <Check className="h-4 w-4 text-fuchsia-500 mr-2" />
        ) : (
          <div className="h-4 w-4 mr-2" />
        )}
        <span className={included ? "" : "text-gray-400"}>{children}</span>
      </div>
    );
  };

const PricingCard = ({ 
  title, 
  price, 
  description, 
  features, 
  ctaText, 
  popular = false,
  yearly = false
}) => (
  <Card className={`flex flex-col h-full ${popular ? 'border-fuchsia-500 shadow-lg relative' : 'border-gray-200'}`}>
    {popular && (
      <div className="absolute -top-4 left-0 right-0 text-center">
        <span className="bg-fuchsia-500 text-white px-4 py-1 rounded-full text-sm font-medium">
          Most Popular
        </span>
      </div>
    )}
    <CardHeader className={`pb-8 ${popular ? 'pt-8' : 'pt-6'}`}>
      <div className="mb-2 font-medium text-fuchsia-600">{title}</div>
      <CardTitle className="flex items-baseline">
        <span className="text-4xl font-bold">${price}</span>
        <span className="ml-1 text-gray-500 font-normal">/{yearly ? 'year' : 'month'}</span>
      </CardTitle>
      <p className="text-gray-500 mt-2">{description}</p>
    </CardHeader>
    <CardContent className="flex-grow">
      <div className="space-y-4">
        {features.map((feature, i) => (
          <FeatureItem 
            key={i}
            included={feature.included}
            tooltip={feature.tooltip}
          >
            {feature.text}
          </FeatureItem>
        ))}
      </div>
    </CardContent>
    <CardFooter className="pt-4 pb-6">
      <Button 
        className={`w-full ${
          popular 
            ? 'bg-fuchsia-600 hover:bg-fuchsia-700 text-white' 
            : 'bg-white hover:bg-gray-50 text-fuchsia-600 border border-fuchsia-300'
        }`}
      >
        {ctaText}
      </Button>
    </CardFooter>
  </Card>
);

const PricingPage = () => {
  const [billingCycle, setBillingCycle] = useState('monthly'); // 'monthly' or 'yearly'
  
  const freeFeatures = [
    { text: '10 minutes of translation per month', included: true },
    { text: '2 languages', included: true },
    { text: 'Standard lip sync', included: true },
    { text: 'File uploads only', included: true },
    { text: 'Basic email support', included: true },
    { text: 'HD video quality', included: false },
    { text: 'YouTube/TikTok integration', included: false },
    { text: 'Priority processing', included: false },
  ];
  
  const creatorFeatures = [
    { text: '2 hours of translation per month', included: true },
    { text: '5 languages', included: true },
    { text: 'Advanced lip sync', included: true, tooltip: 'Higher quality lip sync with improved detail and expression matching' },
    { text: 'File uploads & URL imports', included: true },
    { text: 'Priority email support', included: true },
    { text: 'HD video quality', included: true },
    { text: 'YouTube/TikTok integration', included: true },
    { text: 'Priority processing', included: true },
  ];
  
  const proFeatures = [
    { text: '10 hours of translation per month', included: true },
    { text: 'All languages', included: true },
    { text: 'Premium lip sync', included: true, tooltip: 'Our highest quality lip sync with superior expression matching and emotion preservation' },
    { text: 'All import methods', included: true },
    { text: 'Priority support with 24h response', included: true },
    { text: '4K video quality', included: true },
    { text: 'All platform integrations', included: true },
    { text: 'Express processing', included: true, tooltip: 'Your translations are processed with the highest priority' },
    { text: 'Basic analytics', included: true },
    { text: 'Team access (3 seats)', included: true },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-fuchsia-50 to-white">
      <div className="max-w-6xl mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-fuchsia-600 to-pink-600">
            Simple, Transparent Pricing
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Choose the plan that's right for your content creation needs, with flexible options for creators at any stage.
          </p>
          
          <div className="flex items-center justify-center mt-8">
            <div className="bg-gray-100 p-1 rounded-full inline-flex">
              <button
                className={`px-6 py-2 rounded-full text-sm font-medium ${
                  billingCycle === 'monthly' 
                    ? 'bg-white shadow-sm text-fuchsia-700' 
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                onClick={() => setBillingCycle('monthly')}
              >
                Monthly
              </button>
              <button
                className={`px-6 py-2 rounded-full text-sm font-medium ${
                  billingCycle === 'yearly' 
                    ? 'bg-white shadow-sm text-fuchsia-700' 
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                onClick={() => setBillingCycle('yearly')}
              >
                Yearly
                {billingCycle === 'yearly' ? (
                  <span className="ml-2 bg-green-100 text-green-800 text-xs px-2 py-0.5 rounded-full">
                    Save 20%
                  </span>
                ) : (
                  <span className="ml-2 text-gray-400 text-xs">Save 20%</span>
                )}
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <PricingCard
            title="Free"
            price="0"
            description="Perfect for trying out Magenta's translation capabilities"
            features={freeFeatures}
            ctaText="Get Started"
            yearly={billingCycle === 'yearly'}
          />
          
          <PricingCard
            title="Creator"
            price={billingCycle === 'yearly' ? '23' : '29'}
            description="For content creators building a global audience"
            features={creatorFeatures}
            ctaText="Choose Creator"
            popular={true}
            yearly={billingCycle === 'yearly'}
          />
          
          <PricingCard
            title="Professional"
            price={billingCycle === 'yearly' ? '79' : '99'}
            description="For professionals and teams with higher volume needs"
            features={proFeatures}
            ctaText="Choose Professional"
            yearly={billingCycle === 'yearly'}
          />
        </div>

        <div className="mt-20 bg-white border border-gray-200 rounded-lg shadow-sm p-8">
          <h2 className="text-2xl font-bold mb-4">Enterprise Plans</h2>
          <p className="text-gray-600 mb-6">
            Need custom solutions for your organization? Our Enterprise plans include dedicated support, 
            custom integrations, white-labeling options, and tailored pricing for your specific needs.
          </p>
          <div className="flex flex-col sm:flex-row sm:items-center gap-4">
            <Button className="bg-fuchsia-600 hover:bg-fuchsia-700 text-white">
              Contact Sales
            </Button>
            <span className="text-gray-500">or</span>
            <Button variant="outline" className="border-fuchsia-200 text-fuchsia-600">
              Book a Demo
            </Button>
          </div>
        </div>

        <div className="mt-20">
          <h2 className="text-2xl font-bold mb-6 text-center">Frequently Asked Questions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-bold mb-2">How accurate is the lip sync feature?</h3>
              <p className="text-gray-600">
                Our lip sync technology matches mouth movements to translated speech with high accuracy. 
                The Creator and Professional plans include enhanced versions with better expression preservation.
              </p>
            </div>
            <div>
              <h3 className="font-bold mb-2">Can I upgrade or downgrade my plan?</h3>
              <p className="text-gray-600">
                Yes, you can change your plan at any time. When upgrading, you'll be prorated for the 
                remainder of your billing cycle. When downgrading, changes apply at your next billing date.
              </p>
            </div>
            <div>
              <h3 className="font-bold mb-2">What happens if I exceed my monthly limit?</h3>
              <p className="text-gray-600">
                If you reach your monthly translation limit, you'll have the option to purchase additional 
                minutes or wait until your next billing cycle when your limit resets.
              </p>
            </div>
            <div>
              <h3 className="font-bold mb-2">Do you offer refunds?</h3>
              <p className="text-gray-600">
                We offer a 7-day money-back guarantee on paid plans. If you're not satisfied with our 
                service, contact support within 7 days of your purchase for a full refund.
              </p>
            </div>
          </div>
        </div>
      </div>

      <footer className="bg-white border-t mt-20 py-12">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between">
            <div className="mb-8 md:mb-0">
              <h3 className="text-lg font-bold text-fuchsia-600 mb-4">Magenta AI</h3>
              <p className="text-gray-500 max-w-xs">
                Helping creators reach global audiences through advanced AI translation technology
              </p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-12">
              <div>
                <h4 className="text-sm font-semibold mb-4 text-gray-900">Product</h4>
                <ul className="space-y-2">
                  <li><Link to="/" className="text-gray-500 hover:text-fuchsia-600 text-sm">Features</Link></li>
                  <li><Link to="/pricing" className="text-gray-500 hover:text-fuchsia-600 text-sm">Pricing</Link></li>
                  <li><Link to="/" className="text-gray-500 hover:text-fuchsia-600 text-sm">Roadmap</Link></li>
                </ul>
              </div>
              <div>
                <h4 className="text-sm font-semibold mb-4 text-gray-900">Resources</h4>
                <ul className="space-y-2">
                  <li><Link to="/" className="text-gray-500 hover:text-fuchsia-600 text-sm">Support</Link></li>
                  <li><Link to="/" className="text-gray-500 hover:text-fuchsia-600 text-sm">Documentation</Link></li>
                  <li><Link to="/" className="text-gray-500 hover:text-fuchsia-600 text-sm">Privacy</Link></li>
                  <li><Link to="/" className="text-gray-500 hover:text-fuchsia-600 text-sm">Terms</Link></li>
                </ul>
              </div>
              <div>
                <h4 className="text-sm font-semibold mb-4 text-gray-900">Company</h4>
                <ul className="space-y-2">
                  <li><Link to="/" className="text-gray-500 hover:text-fuchsia-600 text-sm">About</Link></li>
                  <li><Link to="/" className="text-gray-500 hover:text-fuchsia-600 text-sm">Blog</Link></li>
                  <li><Link to="/" className="text-gray-500 hover:text-fuchsia-600 text-sm">Careers</Link></li>
                  <li><Link to="/" className="text-gray-500 hover:text-fuchsia-600 text-sm">Contact</Link></li>
                </ul>
              </div>
            </div>
          </div>
          <div className="border-t border-gray-200 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-500 text-sm">© {new Date().getFullYear()} Magenta AI. All rights reserved.</p>
            <div className="flex space-x-6 mt-4 md:mt-0">
              <a href="#" className="text-gray-400 hover:text-fuchsia-600">
                <span className="sr-only">Twitter</span>
                <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723 10.04 10.04 0 01-3.127 1.184 4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z" />
                </svg>
              </a>
              <a href="#" className="text-gray-400 hover:text-fuchsia-600">
                <span className="sr-only">YouTube</span>
                <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
                </svg>
              </a>
              <a href="#" className="text-gray-400 hover:text-fuchsia-600">
                <span className="sr-only">LinkedIn</span>
                <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default PricingPage;