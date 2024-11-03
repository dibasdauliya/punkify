terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

# Worker deployment
resource "cloudflare_worker_script" "steampunk_transformer" {
  name       = "steampunk-transformer"
  content    = file("${path.module}/src/worker.js")
  account_id = var.cloudflare_account_id
  type       = "module"

  # Enable Workers AI
  plain_text_binding {
    name = "AI_BINDING"
    text = "enabled"
  }
}

# Custom domain for the app
resource "cloudflare_record" "app" {
  zone_id = var.cloudflare_zone_id
  name    = "app"
  value   = cloudflare_worker_script.steampunk_transformer.id
  type    = "CNAME"
}