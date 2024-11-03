# main.tf
terraform {
  required_providers {
    cloudflare = {
      source = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

# Worker deployment
resource "cloudflare_worker_script" "steampunk_transformer" {
  name    = "steampunk-transformer"
  content = file("${path.module}/worker.js")

  # Enable Workers AI
  plain_text_binding {
    name = "AI_BINDING"
    text = "enabled"
  }
}

# Workers AI configuration
resource "cloudflare_workers_ai" "steampunk_ai" {
  account_id = var.cloudflare_account_id
  name       = "steampunk-ai"
  model      = "@cf/stabilityai/stable-diffusion-xl/1.0"
}

# Custom domain for the app
resource "cloudflare_record" "app" {
  zone_id = var.cloudflare_zone_id
  name    = "steampunk-app"
  value   = cloudflare_worker_script.steampunk_transformer.url
  type    = "CNAME"
  proxied = true
}

# Variables
variable "cloudflare_api_token" {
  description = "Cloudflare API Token"
  sensitive   = true
}

variable "cloudflare_account_id" {
  description = "Cloudflare Account ID"
}

variable "cloudflare_zone_id" {
  description = "Cloudflare Zone ID"
}

# Outputs
output "worker_url" {
  value = cloudflare_worker_script.steampunk_transformer.url
}