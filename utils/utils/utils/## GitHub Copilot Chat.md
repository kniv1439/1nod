## GitHub Copilot Chat

- Extension Version: 0.28.4 (prod)
- VS Code: vscode/1.101.2
- OS: Linux
- Remote Name: ssh-remote

## Network

User Settings:
```json
  "github.copilot.advanced.debug.useElectronFetcher": true,
  "github.copilot.advanced.debug.useNodeFetcher": false,
  "github.copilot.advanced.debug.useNodeFetchFetcher": true
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 20.205.243.168 (3 ms)
- DNS ipv6 Lookup: Error (2 ms): getaddrinfo ENOTFOUND api.github.com
- Proxy URL: None (1 ms)
- Electron fetch: Unavailable
- Node.js https: HTTP 403 (369 ms)
- Node.js fetch (configured): HTTP 403 (410 ms)
- Helix fetch: HTTP 403 (709 ms)

Connecting to https://api.individual.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.114.22 (0 ms)
- DNS ipv6 Lookup: Error (3 ms): getaddrinfo ENOTFOUND api.individual.githubcopilot.com
- Proxy URL: None (1 ms)
- Electron fetch: Unavailable
- Node.js https: HTTP 200 (1987 ms)
- Node.js fetch (configured): HTTP 200 (2309 ms)
- Helix fetch: HTTP 200 (2037 ms)

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).
