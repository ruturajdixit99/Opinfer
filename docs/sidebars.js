/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/examples',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'concepts/motion-gating',
        'concepts/adaptive-system',
        'concepts/techniques',
      ],
    },
    {
      type: 'category',
      label: 'Usage Guides',
      items: [
        'guides/vlm-integration',
        'guides/performance-tuning',
        'guides/optimization',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/intro',
        'api/optimized-inference',
        'api/core-classes',
      ],
    },
    {
      type: 'category',
      label: 'Advanced',
      items: [
        'advanced/custom-models',
        'advanced/benchmarking',
      ],
    },
  ],
};

module.exports = sidebars;





