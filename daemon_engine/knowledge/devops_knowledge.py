"""DevOps knowledge base: curated infrastructure and DevOps knowledge.

Integrates content from bregman-arie/devops-exercises (Q&A format) and
jaywcjlove/reference (quick reference sheets). Provides agents with
ready-to-use DevOps and infrastructure knowledge.
"""

from __future__ import annotations

from typing import Any

from daemon_engine.knowledge.knowledge_base import (
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeSource,
)


DEVOPS_KNOWLEDGE: list[dict[str, Any]] = [
    {
        "title": "Docker Fundamentals",
        "content": (
            "Docker containers package applications with their dependencies for "
            "consistent execution across environments.\n\n"
            "Key commands:\n"
            "  docker build -t <tag> .          # Build image from Dockerfile\n"
            "  docker run -p 8080:80 <image>    # Run container with port mapping\n"
            "  docker ps                         # List running containers\n"
            "  docker exec -it <id> bash        # Execute command in container\n"
            "  docker logs <id>                  # View container logs\n"
            "  docker stop <id>                  # Stop container\n"
            "  docker rm <id>                    # Remove container\n"
            "  docker images                     # List images\n"
            "  docker rmi <image>                # Remove image\n\n"
            "Dockerfile best practices:\n"
            "- Use multi-stage builds to reduce image size\n"
            "- Order instructions from least to most frequently changing\n"
            "- Use .dockerignore to exclude unnecessary files\n"
            "- Pin specific base image versions\n"
            "- Run as non-root user\n"
            "- Combine RUN commands to reduce layers"
        ),
        "category": "containerization",
        "tags": ["docker", "containers", "devops", "dockerfile"],
    },
    {
        "title": "Kubernetes Essentials",
        "content": (
            "Kubernetes (K8s) orchestrates containerized applications at scale.\n\n"
            "Core concepts:\n"
            "- Pod: smallest deployable unit, contains one or more containers\n"
            "- Deployment: manages pod replicas and updates\n"
            "- Service: network abstraction for pods\n"
            "- Ingress: HTTP routing to services\n"
            "- ConfigMap: configuration data\n"
            "- Secret: sensitive data (base64 encoded)\n"
            "- Volume: persistent storage\n\n"
            "Key commands (kubectl):\n"
            "  kubectl get pods                  # List pods\n"
            "  kubectl get deployments           # List deployments\n"
            "  kubectl apply -f <file>           # Apply manifest\n"
            "  kubectl delete -f <file>          # Delete resources\n"
            "  kubectl exec -it <pod> -- bash   # Shell into pod\n"
            "  kubectl logs <pod>                # View pod logs\n"
            "  kubectl describe pod <pod>        # Detailed pod info\n"
            "  kubectl scale deploy <d> --replicas=3  # Scale deployment\n"
            "  kubectl rollout status deploy <d> # Check rollout status\n"
            "  kubectl port-forward <pod> 8080:80   # Forward port"
        ),
        "category": "orchestration",
        "tags": ["kubernetes", "k8s", "orchestration", "kubectl"],
    },
    {
        "title": "CI/CD Pipeline Patterns",
        "content": (
            "Continuous Integration/Continuous Deployment automates building, "
            "testing, and deploying code changes.\n\n"
            "Pipeline stages:\n"
            "1. Source: trigger on push/PR\n"
            "2. Build: compile, install dependencies\n"
            "3. Test: unit, integration, e2e tests\n"
            "4. Package: create artifact/image\n"
            "5. Deploy: staging → production\n"
            "6. Verify: health checks, smoke tests\n\n"
            "Best practices:\n"
            "- Keep pipelines fast (< 10 min)\n"
            "- Cache dependencies\n"
            "- Run tests in parallel\n"
            "- Use immutable artifacts\n"
            "- Deploy with blue/green or canary\n"
            "- Always have rollback plan\n\n"
            "Tools: GitHub Actions, GitLab CI, Jenkins, CircleCI, ArgoCD"
        ),
        "category": "cicd",
        "tags": ["ci-cd", "pipeline", "automation", "github-actions"],
    },
    {
        "title": "Linux System Administration",
        "content": (
            "Essential Linux commands for system administration.\n\n"
            "Process management:\n"
            "  ps aux                    # List all processes\n"
            "  top / htop                # Interactive process monitor\n"
            "  kill -9 <pid>             # Force kill process\n"
            "  nohup <cmd> &             # Run in background\n"
            "  systemctl status <svc>    # Check service status\n"
            "  journalctl -u <svc>       # View service logs\n\n"
            "File operations:\n"
            "  find /path -name '*.py'   # Find files by name\n"
            "  grep -r 'pattern' /path   # Search in files\n"
            "  du -sh *                  # Directory sizes\n"
            "  df -h                     # Disk usage\n"
            "  tar -czf archive.tar.gz dir  # Compress\n"
            "  tar -xzf archive.tar.gz   # Extract\n\n"
            "Network:\n"
            "  curl -s URL | jq          # Fetch and parse JSON\n"
            "  wget URL                  # Download file\n"
            "  netstat -tulpn            # List listening ports\n"
            "  ss -tulpn                 # Modern netstat\n"
            "  iptables -L               # Firewall rules\n\n"
            "Permissions:\n"
            "  chmod 755 file            # rwxr-xr-x\n"
            "  chown user:group file     # Change ownership\n"
            "  sudo cmd                  # Run as root"
        ),
        "category": "system",
        "tags": ["linux", "sysadmin", "shell", "commands"],
    },
    {
        "title": "Git Version Control",
        "content": (
            "Git distributed version control system.\n\n"
            "Basic operations:\n"
            "  git init                  # Initialize repo\n"
            "  git clone URL             # Clone repository\n"
            "  git add .                 # Stage all changes\n"
            "  git commit -m 'msg'       # Commit staged changes\n"
            "  git push origin main      # Push to remote\n"
            "  git pull                  # Pull latest changes\n"
            "  git status                # Check working tree\n"
            "  git log --oneline -10     # Recent commits\n\n"
            "Branching:\n"
            "  git branch <name>         # Create branch\n"
            "  git checkout <name>       # Switch branch\n"
            "  git checkout -b <name>    # Create and switch\n"
            "  git merge <branch>        # Merge branch\n"
            "  git rebase main           # Rebase onto main\n"
            "  git branch -d <name>      # Delete branch\n\n"
            "Undo changes:\n"
            "  git checkout -- file      # Discard changes\n"
            "  git reset HEAD~1          # Undo last commit (keep changes)\n"
            "  git reset --hard HEAD~1   # Undo last commit (discard)\n"
            "  git revert <commit>       # Create reverse commit\n\n"
            "Advanced:\n"
            "  git stash                 # Temporarily save changes\n"
            "  git cherry-pick <commit>  # Apply specific commit\n"
            "  git reflog                # Reference log\n"
            "  git bisect                # Binary search for bug"
        ),
        "category": "version-control",
        "tags": ["git", "version-control", "vcs"],
    },
    {
        "title": "Cloud Provider Essentials (AWS)",
        "content": (
            "Amazon Web Services core services.\n\n"
            "Compute:\n"
            "- EC2: Virtual machines\n"
            "- Lambda: Serverless functions\n"
            "- ECS/EKS: Container orchestration\n"
            "- Fargate: Serverless containers\n\n"
            "Storage:\n"
            "- S3: Object storage\n"
            "- EBS: Block storage\n"
            "- EFS: File storage\n\n"
            "Network:\n"
            "- VPC: Virtual network\n"
            "- Route 53: DNS\n"
            "- CloudFront: CDN\n"
            "- Load Balancer: Distribute traffic\n\n"
            "Database:\n"
            "- RDS: Managed relational DB\n"
            "- DynamoDB: NoSQL\n"
            "- ElastiCache: In-memory cache\n\n"
            "CLI: aws s3 ls, aws ec2 describe-instances, aws lambda invoke"
        ),
        "category": "cloud",
        "tags": ["aws", "cloud", "ec2", "s3", "lambda"],
    },
    {
        "title": "Monitoring and Observability",
        "content": (
            "Monitoring ensures system health and performance visibility.\n\n"
            "Three pillars of observability:\n"
            "1. Metrics: numeric measurements over time (CPU, memory, latency)\n"
            "2. Logs: discrete event records\n"
            "3. Traces: request flow across services\n\n"
            "Tools:\n"
            "- Prometheus: metrics collection and alerting\n"
            "- Grafana: visualization dashboards\n"
            "- ELK Stack: Elasticsearch, Logstash, Kibana (logs)\n"
            "- Jaeger/Zipkin: distributed tracing\n"
            "- Datadog: full-stack monitoring\n\n"
            "Best practices:\n"
            "- Define SLIs (indicators), SLOs (objectives), SLAs (agreements)\n"
            "- Alert on symptoms, not causes\n"
            "- Use percentiles (p50, p90, p99) not averages\n"
            "- Implement health checks and readiness probes\n"
            "- Set up alerting runbooks"
        ),
        "category": "monitoring",
        "tags": ["monitoring", "observability", "prometheus", "grafana", "logging"],
    },
    {
        "title": "Infrastructure as Code (IaC)",
        "content": (
            "Infrastructure as Code manages infrastructure through declarative "
            "configuration files.\n\n"
            "Tools:\n"
            "- Terraform: Multi-cloud provisioning (HCL)\n"
            "- Ansible: Configuration management (YAML)\n"
            "- CloudFormation: AWS-native (JSON/YAML)\n"
            "- Pulumi: IaC with real languages (TS, Python, Go)\n\n"
            "Terraform example:\n"
            "  resource 'aws_instance' 'web' {\n"
            "    ami           = 'ami-12345'\n"
            "    instance_type = 't3.micro'\n"
            "    tags = { Name = 'WebServer' }\n"
            "  }\n\n"
            "Ansible playbook example:\n"
            "  - hosts: webservers\n"
            "    tasks:\n"
            "      - name: Install nginx\n"
            "        apt: name=nginx state=present\n"
            "      - name: Start nginx\n"
            "        service: name=nginx state=started\n\n"
            "Best practices:\n"
            "- Version control all IaC\n"
            "- Use modules for reuse\n"
            "- Dry-run before apply\n"
            "- Separate state per environment"
        ),
        "category": "iac",
        "tags": ["terraform", "ansible", "iac", "infrastructure", "cloudformation"],
    },
]


class DevOpsKnowledgeBase:
    """Pre-loaded DevOps knowledge base."""

    def __init__(self, knowledge_base: KnowledgeBase | None = None) -> None:
        self.kb = knowledge_base or KnowledgeBase()
        self._loaded = False

    def load_knowledge(self) -> int:
        if self._loaded:
            return len(self.kb.get_by_source(KnowledgeSource.DEVOPS))
        count = 0
        for item in DEVOPS_KNOWLEDGE:
            self.kb.add_entry(
                title=item["title"],
                content=item["content"],
                source=KnowledgeSource.DEVOPS,
                category=item["category"],
                tags=item["tags"],
                metadata={"type": "devops_reference"},
            )
            count += 1
        self._loaded = True
        return count

    def search(self, query: str, limit: int = 5) -> list[KnowledgeEntry]:
        if not self._loaded:
            self.load_knowledge()
        results = self.kb.search(query, limit=limit, source=KnowledgeSource.DEVOPS)
        return [entry for _, entry in results]

    def get_topic(self, name: str) -> KnowledgeEntry | None:
        if not self._loaded:
            self.load_knowledge()
        for entry in self.kb.get_by_source(KnowledgeSource.DEVOPS):
            if name.lower() in entry.title.lower():
                return entry
        return None

    def list_topics(self) -> list[str]:
        if not self._loaded:
            self.load_knowledge()
        return [e.title for e in self.kb.get_by_source(KnowledgeSource.DEVOPS)]
