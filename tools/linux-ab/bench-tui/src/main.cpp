#include "cpptui.hpp"
#include <memory>
using namespace cpptui;

int main() {
  Theme::set_theme(Theme::Dark());
  App app;
  auto tabs = std::make_shared<Tabs>();
  tabs->add_tab("Runs", std::make_shared<Label>("runs placeholder"));
  tabs->add_tab("System", std::make_shared<Label>("system placeholder"));
  tabs->add_tab("Agents", std::make_shared<Label>("agents placeholder"));
  app.register_key('q', [] { App::quit(); });
  app.run(tabs);
  return 0;
}
