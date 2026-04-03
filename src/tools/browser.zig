//! browser.zig - Browser tool for web search and navigation

const std = @import("std");
const tools_types = @import("types.zig");

const ToolCallRequest = tools_types.ToolCallRequest;
const ToolExecutionResult = tools_types.ToolExecutionResult;
const ToolDefinition = tools_types.ToolDefinition;
const Tool = tools_types.Tool;

/// Search result from web search
pub const SearchResult = struct {
    title: []const u8,
    url: []const u8,
    snippet: []const u8,

    pub fn deinit(self: *SearchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.title);
        allocator.free(self.url);
        allocator.free(self.snippet);
    }
};

/// Page content from opened URL
pub const PageContent = struct {
    url: []const u8,
    title: []const u8,
    content: []const u8,
    links: [][]const u8,

    pub fn deinit(self: *PageContent, allocator: std.mem.Allocator) void {
        allocator.free(self.url);
        allocator.free(self.title);
        allocator.free(self.content);
        for (self.links) |link| allocator.free(link);
        allocator.free(self.links);
    }
};

/// Browser tool implementation
pub const BrowserTool = struct {
    allocator: std.mem.Allocator,
    http_client: std.http.Client,
    search_backend: SearchBackend,
    page_cache: std.StringHashMap(PageContent),
    config: BrowserConfig,

    pub const BrowserConfig = struct {
        timeout_ms: u64 = 30000,
        max_page_size: usize = 1024 * 1024, // 1MB
        user_agent: []const u8 = "Mozilla/5.0 (compatible; zlx-browser/1.0)",
        max_search_results: usize = 10,
    };

    pub const SearchBackend = union(enum) {
        duckduckgo: DuckDuckGoBackend,
        exa: ExaBackend,
        stub: StubBackend,
    };

    pub const DuckDuckGoBackend = struct {
        api_key: ?[]const u8 = null,
    };

    pub const ExaBackend = struct {
        api_key: []const u8,
        base_url: []const u8 = "https://api.exa.ai",
    };

    pub const StubBackend = struct {};

    pub fn init(allocator: std.mem.Allocator, config: BrowserConfig, backend: SearchBackend) !BrowserTool {
        return .{
            .allocator = allocator,
            .http_client = std.http.Client{ .allocator = allocator },
            .search_backend = backend,
            .page_cache = std.StringHashMap(PageContent).init(allocator),
            .config = config,
        };
    }

    pub fn deinit(self: *BrowserTool) void {
        var iter = self.page_cache.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.page_cache.deinit();
        self.http_client.deinit();
    }

    /// Search the web
    pub fn search(self: *BrowserTool, query: []const u8) ![]SearchResult {
        switch (self.search_backend) {
            .duckduckgo => |backend| return try self.searchDuckDuckGo(query, backend),
            .exa => |backend| return try self.searchExa(query, backend),
            .stub => return try self.searchStub(query),
        }
    }

    /// Open a URL and extract content
    pub fn open(self: *BrowserTool, url: []const u8) !PageContent {
        // Check cache first
        if (self.page_cache.get(url)) |cached| {
            return .{
                .url = try self.allocator.dupe(u8, cached.url),
                .title = try self.allocator.dupe(u8, cached.title),
                .content = try self.allocator.dupe(u8, cached.content),
                .links = try self.allocator.dupe([]const u8, cached.links),
            };
        }

        // Fetch page
        var response_body = std.ArrayList(u8).init(self.allocator);
        defer response_body.deinit();

        const uri = try std.Uri.parse(url);
        var server_header_buffer: [8192]u8 = undefined;

        var request = try self.http_client.open(.GET, uri, .{
            .server_header_buffer = &server_header_buffer,
            .headers = .{
                .user_agent = .{ .override = self.config.user_agent },
            },
        });
        defer request.deinit();

        try request.send();
        try request.wait();

        const body = try request.reader().readAllAlloc(self.allocator, self.config.max_page_size);
        defer self.allocator.free(body);

        // Parse HTML to extract content
        const content = try self.extractTextFromHTML(body);
        const title = try self.extractTitleFromHTML(body);

        // Extract links
        const links = try self.extractLinksFromHTML(body, url);

        const page_content = PageContent{
            .url = try self.allocator.dupe(u8, url),
            .title = title,
            .content = content,
            .links = links,
        };

        // Cache the page
        try self.page_cache.put(try self.allocator.dupe(u8, url), page_content);

        return .{
            .url = try self.allocator.dupe(u8, url),
            .title = title,
            .content = content,
            .links = links,
        };
    }

    /// Find text on a page
    pub fn find(self: *BrowserTool, query: []const u8, page_id: []const u8) ![]struct { text: []const u8, context: []const u8 } {
        _ = self;
        _ = page_id;
        // Get cached page
        // Search for query in content
        // Return matches with surrounding context

        var results = std.ArrayList(struct { text: []const u8, context: []const u8 }).init(std.heap.page_allocator);
        // Stub implementation
        try results.append(.{
            .text = try std.heap.page_allocator.dupe(u8, query),
            .context = try std.heap.page_allocator.dupe(u8, "Context around the match..."),
        });
        return results.toOwnedSlice();
    }

    /// Execute tool call
    pub fn executeToolCall(self: *BrowserTool, request: ToolCallRequest, allocator: std.mem.Allocator) !ToolExecutionResult {
        const start_time = std.time.milliTimestamp();

        // Parse JSON arguments
        const Args = struct {
            action: []const u8,
            query: ?[]const u8 = null,
            url: ?[]const u8 = null,
        };

        const args = try std.json.parseFromSlice(Args, allocator, request.arguments, .{});
        defer std.json.parseFree(Args, allocator, args);

        var output = std.ArrayList(u8).init(allocator);
        errdefer output.deinit();

        var success = true;

        if (std.mem.eql(u8, args.action, "search")) {
            if (args.query) |query| {
                const results = try self.search(query);
                defer {
                    for (results) |*r| r.deinit(allocator);
                    allocator.free(results);
                }

                try std.fmt.format(output.writer(), "Search results for '{s}':\n", .{query});
                for (results, 0..) |result, i| {
                    try std.fmt.format(output.writer(), "{d}. {s}\n   {s}\n   {s}\n\n", .{
                        i + 1,
                        result.title,
                        result.url,
                        result.snippet,
                    });
                }
            } else {
                try output.appendSlice("Error: No query provided for search");
                success = false;
            }
        } else if (std.mem.eql(u8, args.action, "open")) {
            if (args.url) |url| {
                const page = try self.open(url);
                defer page.deinit(allocator);

                try std.fmt.format(output.writer(), "Title: {s}\n\n{s}", .{
                    page.title,
                    page.content[0..@min(page.content.len, 2000)],
                });
            } else {
                try output.appendSlice("Error: No URL provided for open");
                success = false;
            }
        } else {
            try std.fmt.format(output.writer(), "Unknown action: {s}", .{args.action});
            success = false;
        }

        const end_time = std.time.milliTimestamp();

        return .{
            .success = success,
            .output = try output.toOwnedSlice(),
            .execution_time_ms = @intCast(end_time - start_time),
        };
    }

    /// Get tool definition
    pub fn getToolDefinition(self: *BrowserTool, allocator: std.mem.Allocator) !ToolDefinition {
        _ = self;
        return .{
            .name = try allocator.dupe(u8, "browser"),
            .description = try allocator.dupe(u8, "Search the web, open URLs, and find text on pages. Actions: search(query), open(url), find(query, page_id)"),
            .parameters = try allocator.dupe(u8,
                \\{{"type": "object", "properties": {{"action": {{"type": "string", "enum": ["search", "open", "find"]}}, "query": {{"type": "string"}}, "url": {{"type": "string"}}}}, "required": ["action"]}}
            ),
        };
    }

    // Private helpers

    fn searchDuckDuckGo(self: *BrowserTool, query: []const u8, backend: DuckDuckGoBackend) ![]SearchResult {
        _ = self;
        _ = backend;
        _ = query;
        // TODO: Implement DuckDuckGo search
        return &[_]SearchResult{};
    }

    fn searchExa(self: *BrowserTool, query: []const u8, backend: ExaBackend) ![]SearchResult {
        _ = self;
        _ = backend;
        _ = query;
        // TODO: Implement Exa search API
        return &[_]SearchResult{};
    }

    fn searchStub(self: *BrowserTool, query: []const u8) ![]SearchResult {
        _ = self;
        // Return stub results for testing
        var results = std.ArrayList(SearchResult).init(std.heap.page_allocator);

        try results.append(.{
            .title = try std.heap.page_allocator.dupe(u8, "Example Result for "),
            .url = try std.heap.page_allocator.dupe(u8, "https://example.com"),
            .snippet = try std.heap.page_allocator.dupe(u8, query),
        });

        return results.toOwnedSlice();
    }

    fn extractTextFromHTML(self: *BrowserTool, html: []const u8) ![]const u8 {
        // Simple HTML to text extraction
        // In production, use a proper HTML parser
        var text = std.ArrayList(u8).init(self.allocator);
        errdefer text.deinit();

        var in_tag = false;
        for (html) |c| {
            if (c == '<') {
                in_tag = true;
            } else if (c == '>') {
                in_tag = false;
            } else if (!in_tag) {
                try text.append(c);
            }
        }

        return text.toOwnedSlice();
    }

    fn extractTitleFromHTML(self: *BrowserTool, html: []const u8) ![]const u8 {
        // Look for <title> tag
        if (std.mem.indexOf(u8, html, "<title>")) |start| {
            const title_start = start + 7;
            if (std.mem.indexOfPos(u8, html, title_start, "</title>")) |end| {
                return try self.allocator.dupe(u8, html[title_start..end]);
            }
        }
        return try self.allocator.dupe(u8, "Untitled");
    }

    fn extractLinksFromHTML(self: *BrowserTool, html: []const u8, base_url: []const u8) ![][]const u8 {
        _ = base_url;
        _ = html;
        // TODO: Extract href attributes from <a> tags
        return try self.allocator.alloc([]const u8, 0);
    }

    // Tool interface implementation

    pub fn asTool(self: *BrowserTool) Tool {
        return .{
            .vtable = &.{
                .execute = toolExecute,
                .getDefinition = toolGetDefinition,
                .deinit = toolDeinit,
            },
            .ptr = self,
        };
    }

    fn toolExecute(ctx: *anyopaque, request: ToolCallRequest, allocator: std.mem.Allocator) anyerror!ToolExecutionResult {
        const self = @as(*BrowserTool, @ptrCast(@alignCast(ctx)));
        return self.executeToolCall(request, allocator);
    }

    fn toolGetDefinition(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!ToolDefinition {
        const self = @as(*BrowserTool, @ptrCast(@alignCast(ctx)));
        return self.getToolDefinition(allocator);
    }

    fn toolDeinit(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self = @as(*BrowserTool, @ptrCast(@alignCast(ctx)));
        self.deinit();
        allocator.destroy(self);
    }
};
