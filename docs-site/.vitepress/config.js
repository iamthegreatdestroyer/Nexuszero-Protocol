import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Nexuszero Protocol',
  description: 'Quantum-resistant zero-knowledge proof system based on lattice cryptography',
  ignoreDeadLinks: 'localhostLinks',
  
  themeConfig: {
    logo: '/logo.svg',
    
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API Reference', link: '/api/client' },
      { text: 'Examples', link: '/examples/age-verification' },
      { 
        text: 'GitHub',
        link: 'https://github.com/iamthegreatdestroyer/Nexuszero-Protocol'
      }
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Introduction',
          items: [
            { text: 'What is Nexuszero?', link: '/guide/what-is-nexuszero' },
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'Installation', link: '/guide/installation' },
          ]
        },
        {
          text: 'Core Concepts',
          items: [
            { text: 'Zero-Knowledge Proofs', link: '/guide/zero-knowledge-proofs' },
            { text: 'Range Proofs', link: '/guide/range-proofs' },
            { text: 'Security Levels', link: '/guide/security-levels' },
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'NexuszeroClient', link: '/api/client' },
            { text: 'ProofBuilder', link: '/api/proof-builder' },
            { text: 'Crypto Functions', link: '/api/crypto' },
            { text: 'Types', link: '/api/types' },
            { text: 'Error Handling', link: '/api/errors' },
          ]
        }
      ],
      '/examples/': [
        {
          text: 'Examples',
          items: [
            { text: 'Age Verification', link: '/examples/age-verification' },
            { text: 'Salary Range', link: '/examples/salary-range' },
            { text: 'Balance Check', link: '/examples/balance-check' },
            { text: 'Custom Proofs', link: '/examples/custom-proofs' },
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/iamthegreatdestroyer/Nexuszero-Protocol' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2025 Nexuszero Protocol'
    },

    search: {
      provider: 'local'
    }
  }
})
