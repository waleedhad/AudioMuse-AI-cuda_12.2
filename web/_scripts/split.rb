require 'fileutils'

# --- Configuration ---
# The script assumes it's run from the 'web' directory.
SOURCE_FILE = '../README.md'
OUTPUT_DIR = './_sections'
SPLIT_MARKER = "\n## " # We split the content at every H2 heading

# --- Script Logic ---
puts "Starting README split..."

# Ensure output directory exists and is empty
FileUtils.rm_rf(OUTPUT_DIR) if File.directory?(OUTPUT_DIR)
FileUtils.mkdir_p(OUTPUT_DIR)

# Read the source README.md file
full_content = File.read(SOURCE_FILE)

# Remove the YAML front matter from the root README, we don't need it.
content_without_front_matter = full_content.sub(/---.*?---/m, '').strip

# Split the content by H2 headings
sections = content_without_front_matter.split(SPLIT_MARKER)

# The first chunk is the intro content before the first H2
intro_content = sections.shift.strip

# Create the index page from the intro content
index_front_matter = <<~FM
  ---
  layout: default
  title: Home
  order: 0
  ---
FM
File.write(File.join(OUTPUT_DIR, 'index.md'), index_front_matter + intro_content)
puts "Created index.md"

# Process each subsequent section
sections.each_with_index do |section_content, i|
  # The heading is the first line of the section content
  lines = section_content.lines
  heading = lines.first.strip
  content_body = lines[1..-1].join

  # Create a URL-friendly filename (a "slug")
  slug = heading.downcase.strip.gsub(' ', '-').gsub(/[^\w-]/, '')
  filename = "#{i + 1}-#{slug}.md"

  # Create the YAML front matter for the new page file
  page_front_matter = <<~FM
    ---
    layout: default
    title: #{heading}
    order: #{i + 1}
    ---
  FM

  # Write the new Markdown file
  File.write(File.join(OUTPUT_DIR, filename), page_front_matter + "## #{heading}\n" + content_body)
  puts "Created #{filename} with title '#{heading}'"
end

puts "Splitting complete. Total pages created: #{sections.length + 1}"
